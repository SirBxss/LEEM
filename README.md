# Lane-Estimation Error Modelling

This package implements the data and modelling pipeline for the lane-estimation error modelling thesis. It creates reproducible, variable-length sequences with a common data contract for three model families:

1. conditional multivariate Gaussian;
2. autoregressive input-output hidden Markov model (AIOHMM);
3. recurrent conditional generative adversarial network (RC-GAN).

The synthetic data are controlled implementation and capability tests. They are **not evidence of real BMW sensor behaviour** and must not be used for final model ranking or real-world claims.

The package also contains the common preprocessing/model interfaces and all three planned model families for synthetic and later BMW-derived datasets. The main source layout is:

```text
src/lane_error_modeling/
├── data/
│   ├── synthetic/
│   └── preprocessing/
├── evaluation/
└── models/
    ├── aiohmm/
    ├── base.py
    ├── gaussian/
    └── rcgan/
```

## Data contract

At time $t$, the condition vector is

$$
X_t=[v_t,\bar\kappa_t,\Delta\kappa_t,w_{\mathrm{lane},t},q_{\mathrm{mark},t},q_{\mathrm{env},t}]^\mathsf T.
$$

The output is a signed lateral error profile at 21 look-ahead stations:

$$
Y_t=[e_d(t,0),e_d(t,5),\ldots,e_d(t,100)]^\mathsf T.
$$

Errors are measured along the reference-path normal. Arrays are padded only at serialization/batching time, and `valid_mask` distinguishes valid values from unavailable look-ahead stations and padding.

## Installation

From this directory:

```bash
python -m pip install -e .
```

The core runtime dependency is NumPy. Install the optional evaluation plots with:

```bash
python -m pip install -e ".[evaluation]"
```

RC-GAN additionally requires PyTorch:

```bash
python -m pip install -e ".[evaluation,rcgan]"
```

## Generate the verified smoke dataset

```bash
generate-lane-error-data \
  --config configs/synthetic_smoke.json \
  --output outputs/synthetic_smoke
```

Re-running into a non-empty output directory is rejected. A generated directory can be replaced explicitly:

```bash
generate-lane-error-data \
  --config configs/synthetic_smoke.json \
  --output outputs/synthetic_smoke \
  --overwrite
```

The overwrite operation is accepted only if the target contains the generator's `manifest.json`.

## Generate the prototype dataset

```bash
generate-lane-error-data \
  --config configs/synthetic_prototype.json \
  --output outputs/synthetic_prototype
```

The configured train/validation/test counts are generated independently for **each** of the three scenarios. Use the smoke configuration for development and continuous tests; generate the prototype only for model experiments.

## Output layout

```text
outputs/synthetic_smoke/
├── manifest.json
├── conditional_gaussian/
│   ├── train.npz
│   ├── validation.npz
│   └── test.npz
├── latent_autoregressive/
│   └── ...
└── nonlinear_heavy_tailed/
    └── ...
```

Each `.npz` contains:

| Array | Shape | Description |
|---|---:|---|
| `sequence_ids` | $[B]$ | Stable scenario/split/index identifiers |
| `sequence_seeds` | $[B]$ | Independently reproducible sequence seeds |
| `lengths` | $[B]$ | Original lengths before padding |
| `conditions` | $[B,T,6]$ | Six physical/scene conditions |
| `errors` | $[B,T,21]$ | Signed lateral errors in metres |
| `valid_mask` | $[B,T,21]$ | Valid target values |
| `conditional_mean` | $[B,T,21]$ | Oracle DGP mean, unavailable for real data |
| `latent_state` | $[B,T]$ | Oracle regime/burst state, unavailable for real data |
| `reference_curvature` | $[B,T,21]$ | Reference curvature profiles |
| `reference_heading` | $[B,T,21]$ | Integrated reference headings |
| `reference_xy` | $[B,T,21,2]$ | Reference paths in ego-local coordinates |
| `s_grid_m` | $[21]$ | Look-ahead stations in metres |

Load a split without enabling pickle:

```python
from lane_error_modeling.data.synthetic.io import load_dataset

dataset = load_dataset(
    "outputs/synthetic_smoke/conditional_gaussian/train.npz"
)
print(dataset.conditions.shape, dataset.errors.shape)
```

## Train-only standardization and sequence batches

Convert a serialized split into the common model-facing dataset, fit the transform on training data only, and create complete-sequence batches:

```python
from lane_error_modeling.data.preprocessing import (
    SequenceDataset,
    SequenceStandardizer,
    iter_sequence_batches,
)
from lane_error_modeling.data.synthetic.io import load_dataset
from lane_error_modeling.data.synthetic.schema import FEATURE_NAMES

raw = load_dataset(
    "outputs/synthetic_smoke/conditional_gaussian/train.npz"
)
train = SequenceDataset.from_arrays(
    sequence_ids=raw.sequence_ids,
    conditions=raw.conditions,
    errors=raw.errors,
    valid_mask=raw.valid_mask,
    lengths=raw.lengths,
    feature_names=FEATURE_NAMES,
    s_grid_m=raw.s_grid_m,
)

standardizer = SequenceStandardizer().fit(
    train.conditions,
    train.errors,
    train.valid_mask,
    train.lengths,
    split_name="train",
    feature_names=train.feature_names,
    s_grid_m=train.s_grid_m,
)
standardizer.save("outputs/standardization.json")
standardized_train = train.standardized_copy(standardizer)

for batch in iter_sequence_batches(
    standardized_train,
    batch_size=4,
    shuffle=True,
    seed=20260710,
):
    print(batch.conditions.shape, batch.errors.shape)
```

Conditions are standardized feature-wise over active training frames. Errors are standardized station-wise using only values selected by `valid_mask`. Padding and unavailable targets remain zero. Samples must be inverse-transformed before any physical-unit metric or planner experiment.

The manifest stores the complete configuration, feature names, generator/schema versions, SHA-256 checksum, file size, error quantiles, feature ranges, valid fraction, and latent-state counts for every split.

## Conditional multivariate Gaussian baseline

The baseline estimates a masked linear conditional mean at every look-ahead station and a full residual spatial covariance. Pairwise observed covariance, diagonal shrinkage, and a positive-definite projection make fitting compatible with variable visible range.

```python
from lane_error_modeling.models import (
    ConditionalMultivariateGaussian,
    GaussianConfig,
)

model = ConditionalMultivariateGaussian(GaussianConfig())
# Create standardized validation/test datasets with this same standardizer.
report = model.fit(standardized_train, standardized_validation)
samples = model.sample(
    standardized_test.conditions,
    standardized_test.lengths,
    n_samples=100,
    seed=20260713,
    valid_mask=standardized_test.valid_mask,
)
log_probability = model.log_probability(standardized_test)
model.save("outputs/models/gaussian.npz")
```

The model expects the same frozen training standardizer for fitting, scoring, sampling interpretation, and physical-unit inverse transformation. See [Conditional multivariate Gaussian baseline](docs/conditional_multivariate_gaussian.md) for its equations, missing-data estimator, assumptions, and verification protocol.

## Autoregressive input-output HMM

The AIOHMM adds a validation-selected latent state, input-dependent state transitions, station-wise AR(1) memory, and a full state-specific spatial covariance. It is the temporal statistical model between the Gaussian baseline and the later RC-GAN.

```python
from lane_error_modeling.models import (
    AIOHMMConfig,
    AutoregressiveInputOutputHMM,
)

model = AutoregressiveInputOutputHMM(AIOHMMConfig(n_states=4))
report = model.fit(standardized_train, standardized_validation)
samples = model.sample(
    standardized_test.conditions,
    standardized_test.lengths,
    n_samples=100,
    seed=20260715,
    valid_mask=standardized_test.valid_mask,
)
log_probability = model.log_probability(standardized_test)
model.save("outputs/models/aiohmm.npz")
```

Missing current target dimensions are marginalized. A missing previous station value contributes no AR term, equivalent to its standardized training mean. See [Autoregressive input-output HMM](docs/autoregressive_input_output_hmm.md) for the full multivariate adaptation, generalized-EM estimator, assumptions, and diagnostics.

## Recurrent conditional GAN

The third model follows Arnelid et al.'s RC-GAN: a separate one-layer noise LSTM,
a deep condition LSTM, condition skip connections, and a recurrent conditional
discriminator. LEEM adds target masks and padded-sequence loss masks while
retaining full profile generation.

```python
from lane_error_modeling.models import RCGANConfig, RecurrentConditionalGAN

model = RecurrentConditionalGAN(RCGANConfig())
report = model.fit(standardized_train, standardized_validation)
samples = model.sample(
    standardized_test.conditions,
    standardized_test.lengths,
    n_samples=100,
    seed=20260720,
    valid_mask=standardized_test.valid_mask,
)
model.save("outputs/models/rcgan_model.npz")
```

RC-GAN has no tractable normalized likelihood. Declared learning-rate/seed
candidates are first checked by a validation-only conditional-diversity guard,
then selected by physical-unit, dimension-normalized Energy Score. See [Recurrent
conditional GAN](docs/recurrent_conditional_gan.md) for the architecture,
objectives, paper fidelity, LEEM adaptations, and experiment protocol.

## Common evaluation and model experiments

Phase 5 selects Gaussian hyperparameters using validation data only and evaluates the held-out test split once. Common sample-based metrics are computed in metres and are designed to remain applicable to AIOHMM and RC-GAN.

Run the smoke experiment:

```bash
python scripts/run_gaussian_experiment.py \
  --config configs/gaussian_experiment_smoke.json \
  --output outputs/experiments/gaussian_smoke
```

After the smoke experiment passes, generate the prototype dataset and run:

```bash
python scripts/run_gaussian_experiment.py \
  --config configs/gaussian_experiment_prototype.json \
  --output outputs/experiments/gaussian_prototype
```

The output contains the frozen standardizer/model, validation candidate table, physical-unit test metrics, synthetic oracle diagnostics, plots, and SHA-256 provenance. Generated experiment outputs remain excluded from Git. See [Common evaluation and Gaussian experiment protocol](docs/evaluation_protocol.md) for metric equations and interpretation rules.

After the Gaussian experiment, run AIOHMM with the same data and common evaluator:

```bash
python scripts/run_aiohmm_experiment.py \
  --config configs/aiohmm_experiment_smoke.json \
  --output outputs/experiments/aiohmm_smoke
```

The AIOHMM search compares hidden-state counts and deterministic initialization restarts using validation NLL. Its test split is not loaded until selection is frozen. After the smoke gate passes:

```bash
python scripts/run_aiohmm_experiment.py \
  --config configs/aiohmm_experiment_prototype.json \
  --output outputs/experiments/aiohmm_prototype
```

Install the deep-learning extra and run the RC-GAN software gate:

```bash
python -m pip install -e ".[evaluation,rcgan]"
python scripts/run_rcgan_experiment.py \
  --config configs/rcgan_experiment_smoke.json \
  --output outputs/experiments/rcgan_smoke
```

The smoke run verifies software only. Before the longer prototype, generate the
small one-scenario stability-pilot dataset and run its three declared learning
rates:

```bash
generate-lane-error-data \
  --config configs/synthetic_rcgan_pilot.json \
  --output outputs/synthetic_rcgan_pilot

python scripts/run_rcgan_experiment.py \
  --config configs/rcgan_experiment_pilot.json \
  --output outputs/experiments/rcgan_pilot
```

Only proceed when at least one candidate passes the predeclared diversity guard
and the epoch history does not show non-finite losses or persistent gradient
clipping. Freeze the learning rate selected by this pilot in the prototype
configuration before running the longer two-restart, three-scenario experiment:

```bash
python scripts/run_rcgan_experiment.py \
  --config configs/rcgan_experiment_prototype.json \
  --output outputs/experiments/rcgan_prototype
```

Existing persisted results can be upgraded with finite-ensemble interval
metadata and compared without retraining either model:

```bash
python scripts/upgrade_evaluation_results.py \
  --root results/synthetic \
  --write

python scripts/compare_experiments.py \
  --baseline results/synthetic/gaussian_prototype \
  --candidate results/synthetic/aiohmm_prototype \
  --output results/synthetic/gaussian_vs_aiohmm
```

The comparison first checks that both models used the same scenarios,
observations, station grid, and evaluation references. The output is written
as deterministic JSON, CSV, and Markdown tables.

## Scientific safeguards

- Train, validation, and test sequences use disjoint deterministic seeds.
- A seed is derived from master seed, scenario, split, and sequence index; generation order does not affect a sequence.
- Feature normalization must later be fitted on training data only.
- Invalid targets are masked rather than imputed with zeros.
- The three DGP scenarios are trained/evaluated separately; they are not pooled into one artificial benchmark.
- Oracle means and latent states may be used only for implementation diagnostics, never as model inputs.
- Final real-data splits must be made by complete drive or scenario, never random neighbouring frames.

## Documentation

The mathematical definitions, parameter choices, assumptions, validation protocol, and limitations are documented in:

- [Synthetic data generation methodology](docs/synthetic_data_generation.md)
- [Preprocessing and common model contract](docs/preprocessing_and_model_contract.md)
- [Conditional multivariate Gaussian baseline](docs/conditional_multivariate_gaussian.md)
- [Autoregressive input-output HMM](docs/autoregressive_input_output_hmm.md)
- [Recurrent conditional GAN](docs/recurrent_conditional_gan.md)
- [Phase 6 AIOHMM smoke results](docs/phase6_smoke_results.md)
- [Common evaluation and model experiment protocol](docs/evaluation_protocol.md)
- [Phase 6.1 evaluation reporting and comparison](docs/phase6_1_evaluation_reporting.md)

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover configuration validation, path geometry, signed error recovery, deterministic generation, split independence, masks, serialization, manifest integrity, intended scenario properties, exact HMM inference, AIOHMM and RC-GAN fitting/sampling/persistence, and leakage-safe model selection.

The core end-to-end scientific checks can also be run directly:

```bash
PYTHONPATH=src python scripts/verify_synthetic_pipeline.py
PYTHONPATH=src python scripts/verify_preprocessing_pipeline.py
PYTHONPATH=src python scripts/verify_gaussian_model.py
PYTHONPATH=src python scripts/run_gaussian_experiment.py --config configs/gaussian_experiment_smoke.json --output outputs/experiments/gaussian_smoke
PYTHONPATH=src python scripts/run_aiohmm_experiment.py --config configs/aiohmm_experiment_smoke.json --output outputs/experiments/aiohmm_smoke
PYTHONPATH=src python scripts/run_rcgan_experiment.py --config configs/rcgan_experiment_smoke.json --output outputs/experiments/rcgan_smoke
```
