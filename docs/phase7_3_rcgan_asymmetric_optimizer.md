# Phase 7.3: RC-GAN Asymmetric-Optimizer Micro-Pilot

## 1. Decision from Phase 7.2

Phase 7.2 produced a valid `stability_failed` result: all three candidates
completed training, none passed every validation-only gate, and the held-out
test split remained unopened.

| Shared learning rate | Diversity ratio | 90% coverage/reference | Late mean generator clipping | Validation Energy Score | Interpretation |
|---:|---:|---:|---:|---:|---|
| $10^{-5}$ | 0.0741 | 0.1080 | 0.0000 | 0.0556 | Numerically stable but conditionally collapsed |
| $3\times10^{-5}$ | 0.0777 | 0.0985 | 0.3646 | 0.0653 | Marginal tails improved, but conditional spread remained collapsed and final clipping increased |
| $5\times10^{-5}$ | 0.0945 | 0.0355 | 0.7474 | 0.1659 | Diversity increased slightly through unstable, inaccurate generation |

Learning-rate scaling of both adversaries together is therefore not an adequate
repair. The discriminator becomes too strong as the shared rate increases,
while the generator still responds too weakly to latent noise.

Phase 7.3 is the final bounded, paper-architecture-preserving stabilization
attempt. It is not an open-ended hyperparameter search, a fourth thesis model,
or authorization to run the full three-scenario prototype.

## 2. Scientific hypothesis

The architecture and binary adversarial objectives remain unchanged. Only the
optimizer time scales are separated:

$$
\theta_G \leftarrow
\operatorname{Adam}\!\left(\theta_G,
\nabla_{\theta_G}\mathcal L_G,\eta_G\right),
$$

$$
\theta_D \leftarrow
\operatorname{Adam}\!\left(\theta_D,
\nabla_{\theta_D}\mathcal L_D,\eta_D\right).
$$

The Phase 7.2 results motivate fixing $\eta_D=10^{-5}$ while testing two
generator rates:

$$
(\eta_G,\eta_D)\in
\left\{(3\times10^{-5},10^{-5}),
(5\times10^{-5},10^{-5})\right\}.
$$

This asks one precise question: can the generator learn stronger latent-noise
dependence without allowing the discriminator instability seen when both
networks used the higher rate?

The optional `discriminator_learning_rate` is a documented LEEM stabilization
adaptation. When it is absent, both optimizers use the original shared
`learning_rate`, preserving all earlier experiment configurations and persisted
models.

## 3. Fixed micro-pilot design

Phase 7.3 reuses `outputs/synthetic_rcgan_pilot`; no data are regenerated.

| Setting | Value |
|---|---:|
| Scenario | `conditional_gaussian` only |
| Training/validation/test sequences | 128 / 32 / 32 |
| Initialization seeds | one: 20260717 |
| Epochs | 4 |
| Batch size | 1 |
| Generator rates | $3\times10^{-5}$, $5\times10^{-5}$ |
| Discriminator rate | $10^{-5}$ |
| Validation selection draws | 64 |
| Per-epoch diagnostic draws | 16 |
| Ranking metric after gating | dimension-normalized Energy Score |

Four batch-size-one epochs correspond to 512 sequence updates per candidate.
The earlier Phase 7.1 run used 12 batch-size-four epochs, or 384 sequence
updates. Phase 7.3 is therefore not a deliberately under-trained shortcut.

## 4. Additional noise-path diagnostics

The generator has a separate noise LSTM with parameters $\theta_z$, a context
LSTM with parameters $\theta_x$, and a dense output head. Before global gradient
clipping, every update now records the three group norms

$$
g_z=\lVert\nabla_{\theta_z}\mathcal L_G\rVert_2,
\qquad
g_x=\lVert\nabla_{\theta_x}\mathcal L_G\rVert_2,
\qquad
g_h=\lVert\nabla_{\theta_h}\mathcal L_G\rVert_2,
$$

and the epoch ratio

$$
r_{z/x}=\frac{\sum g_z}{\max(\sum g_x,10^{-12})}.
$$

A very small $r_{z/x}$ supports the diagnosis that adversarial training mainly
updates the condition path while neglecting the latent-noise path. A nonzero
gradient alone does not prove useful stochastic generation, so the existing
output-diversity metric remains mandatory.

In addition to the global conditional-diversity ratio, each epoch records the
minimum, median, and maximum station-wise generated-to-observed standard
deviation ratios. This detects a generator that uses noise only at a small
subset of the 21 look-ahead stations.

## 5. Strengthened stability gate

All decisions remain validation-only. A candidate must pass every enabled
criterion before Energy-Score ranking.

### 5.1 Diversity and conditional coverage

- final global conditional-diversity ratio: at least 0.10;
- 90% empirical interval coverage: at least 50% of the 64-draw finite-ensemble
  reference.

### 5.2 Mean and worst-late optimization behavior

For the final three epochs, both the mean and the worst single epoch are checked:

- generator clipped-update fraction: at most 0.50;
- discriminator real/fake probability gap: at most 0.75.

The worst-late checks prevent an unstable final epoch from being hidden by two
earlier stable epochs. This specifically corrects the misleading individual
passes observed in Phase 7.2.

### 5.3 Two-sided tail check

Let

$$
r_{\mathrm{tail}}=
\frac{P_{\mathrm{generated}}(|e|>q_{0.95}^{\mathrm{train}})}
{P_{\mathrm{observed}}(|e|>q_{0.95}^{\mathrm{train}})}.
$$

Phase 7.3 requires

$$
0.20\leq r_{\mathrm{tail}}\leq5.0.
$$

The lower bound rejects missing tails. The new upper bound rejects explosive
marginal tails such as the Phase 7.2 value of 7.33. These bounds are permissive
engineering guards, not calibration claims.

## 6. Leakage safety and portable provenance

Training, validation gating, and Energy-Score ranking occur before test loading.
If both candidates fail, the runner persists all histories and validation
diagnostics with status `stability_failed`; it does not load or record provenance
for the test split.

New result artifacts store project-relative paths such as
`outputs/synthetic_rcgan_pilot/...` instead of public absolute workstation paths.
External temporary configurations used by tests retain their absolute path only
when no project-relative representation exists.

## 7. Commands

From the PyCharm PowerShell terminal at the LEEM project root:

```powershell
python -m pip install -e ".[evaluation,rcgan]"
python -m unittest discover -s tests -v

python scripts/run_rcgan_experiment.py `
  --config configs/rcgan_experiment_pilot_v3.json `
  --output results/synthetic/rcgan_pilot_v3
```

The command exits with code 0 for `status=passed` and code 2 for
`status=stability_failed`. Both outcomes preserve an auditable result directory.

Push the result in either case:

```powershell
git add results/synthetic/rcgan_pilot_v3
git commit -m "add Phase 7.3 RC-GAN asymmetric-optimizer results"
git push origin main
```

## 8. Terminal decision rule

If at least one candidate passes, freeze the selected generator/discriminator
rates in the prototype configuration before any three-scenario run.

If both candidates fail, stop synthetic RC-GAN tuning. Retain RC-GAN as the
third thesis model and report the paper-based 21-dimensional LEEM adaptation as
unstable or under-dispersed under the fixed experimental budget. Do not replace
it with a fourth model, weaken the gates retrospectively, or inspect the Phase
7.3 test split. A scientifically controlled negative result remains part of the
three-model thesis comparison.

## 9. Files in Phase 7.3

New files:

- `configs/rcgan_experiment_pilot_v3.json`
- `docs/phase7_3_rcgan_asymmetric_optimizer.md`

Updated files:

- `README.md`
- `docs/evaluation_protocol.md`
- `docs/phase7_2_rcgan_stabilization.md`
- `docs/recurrent_conditional_gan.md`
- `src/lane_error_modeling/evaluation/config.py`
- `src/lane_error_modeling/evaluation/rcgan_experiment.py`
- `src/lane_error_modeling/models/rcgan/config.py`
- `src/lane_error_modeling/models/rcgan/model.py`
- `tests/test_rcgan_experiment.py`
- `tests/test_rcgan_model.py`
