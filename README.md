# Lane-Estimation Error Modelling

This package implements the synthetic-data phase of the lane-estimation error modelling thesis. It creates reproducible, variable-length sequences with a common data contract for three later model families:

1. conditional multivariate Gaussian;
2. autoregressive input-output hidden Markov model (AIOHMM);
3. recurrent conditional generative adversarial network (RC-GAN).

The synthetic data are controlled implementation and capability tests. They are **not evidence of real BMW sensor behaviour** and must not be used for final model ranking or real-world claims.

## Data contract

At time t, the condition vector X_t contains:

vehicle speed v_t mean curvature kappa_bar_t curvature difference delta_kappa_t lane width w_lane,t lane-marking quality q_mark,t environmental quality q_env,t

In compact form:

X_t = [v_t, kappa_bar_t, delta_kappa_t, w_lane,t, q_mark,t, q_env,t]^T

The output Y_t is a signed lateral-error profile evaluated at 21 look-ahead stations from 0 to **100** metres in 5-metre intervals.

In compact form:

Y_t = [e_d(t, 0), e_d(t, 5), ..., e_d(t, **100**)]^T

Errors are measured along the reference-path normal. Arrays are padded only at serialization/batching time, and `valid_mask` distinguishes valid values from unavailable look-ahead stations and padding.

## Installation

From this directory:

```bash
python -m pip install -e .
```

The only runtime dependency is NumPy. The test suite uses the Python standard library.

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
| `sequence_ids` | \([B]\) | Stable scenario/split/index identifiers |
| `sequence_seeds` | \([B]\) | Independently reproducible sequence seeds |
| `lengths` | \([B]\) | Original lengths before padding |
| `conditions` | \([B,T,6]\) | Six physical/scene conditions |
| `errors` | \([B,T,21]\) | Signed lateral errors in metres |
| `valid_mask` | \([B,T,21]\) | Valid target values |
| `conditional_mean` | \([B,T,21]\) | Oracle DGP mean, unavailable for real data |
| `latent_state` | \([B,T]\) | Oracle regime/burst state, unavailable for real data |
| `reference_curvature` | \([B,T,21]\) | Reference curvature profiles |
| `reference_heading` | \([B,T,21]\) | Integrated reference headings |
| `reference_xy` | \([B,T,21,2]\) | Reference paths in ego-local coordinates |
| `s_grid_m` | \([21]\) | Look-ahead stations in metres |

Load a split without enabling pickle:

```python
from lane_error_modeling.data.synthetic.io import load_dataset

dataset = load_dataset(
    "outputs/synthetic_smoke/conditional_gaussian/train.npz"
)
print(dataset.conditions.shape, dataset.errors.shape)
```

The manifest stores the complete configuration, feature names, generator/schema versions, SHA-256 checksum, file size, error quantiles, feature ranges, valid fraction, and latent-state counts for every split.

## Scientific safeguards

- Train, validation, and test sequences use disjoint deterministic seeds.
- A seed is derived from master seed, scenario, split, and sequence index; generation order does not affect a sequence.
- Feature normalization must later be fitted on training data only.
- Invalid targets are masked rather than imputed with zeros.
- The three DGP scenarios are trained/evaluated separately; they are not pooled into one artificial benchmark.
- Oracle means and latent states may be used only for implementation diagnostics, never as model inputs.
- Final real-data splits must be made by complete drive or scenario, never random neighbouring frames.

## Documentation

The mathematical definitions, parameter choices, assumptions, validation protocol, and limitations are documented in [Synthetic data generation methodology](docs/synthetic_data_generation.md).

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover configuration validation, path geometry, signed error recovery, deterministic generation, split independence, masks, serialization, manifest integrity, and intended scenario properties.

The core end-to-end scientific checks can also be run directly:

```bash
PYTHONPATH=src python scripts/verify_synthetic_pipeline.py
```
