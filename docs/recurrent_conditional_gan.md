# Recurrent Conditional Generative Adversarial Network

## 1. Purpose

RC-GAN is the third and most flexible LEEM thesis model. Its purpose is to learn
an implicit conditional distribution over complete, temporally coherent
21-station lane-estimation error profiles without assuming Gaussian residuals or
a finite number of latent regimes.

The implementation is based on Arnelid et al., *Recurrent Conditional
Generative Adversarial Networks for Autonomous Driving Sensor Modelling*
([DOI: 10.1109/ITSC.2019.8916999](https://doi.org/10.1109/ITSC.2019.8916999)).
It preserves the paper's defining network separation and adversarial objective.
The original paper models another autonomous-driving sensor error, whereas LEEM
models a vector of lateral lane errors. This is therefore a documented
multivariate adaptation, not a claim of exact result reproduction.

Synthetic experiments verify implementation and model capability only. They do
not show that real BMW lane-estimation errors follow the synthetic processes or
that RC-GAN will outperform the other models on BMW data.

## 2. Core idea

At every active time step, the generator receives:

- standardized conditions $\mathbf x_t\in\mathbb R^F$;
- independent latent noise $\mathbf z_t\in\mathbb R^L$.

Noise and conditions are deliberately processed by separate recurrent paths.
Arnelid et al. introduced this separation because noise tended to disappear when
both inputs shared one deep recurrent network. The generator also skip-connects
the raw conditions to its dense head. This makes current scene information
available without forcing it through recurrent memory.

The discriminator receives either a real or generated error profile together
with the same conditions. It uses a recurrent network and condition skip
connection to issue one real/fake logit per time step. The two networks are
trained alternately.

## 3. Mathematics

### 3.1 Generator

For a LEEM sequence, $F=6$, $K=21$, and the prototype latent dimension is
$L=32$. The two generator states are

$$
\mathbf h_t^{z}=\operatorname{LSTM}_{z}
(\mathbf z_t,\mathbf h_{t-1}^{z}),
$$

$$
\mathbf h_t^{x}=\operatorname{LSTM}_{x}
(\mathbf x_t,\mathbf h_{t-1}^{x}).
$$

The noise path has one LSTM layer. The context path has two layers in the final
conference-paper architecture. Their outputs and the raw condition skip are
concatenated:

$$
\mathbf u_t=[\mathbf h_t^{z},\mathbf h_t^{x},\mathbf x_t].
$$

The generated standardized lane-error profile is

$$
\widehat{\mathbf y}_t=
W_o\,\phi(W_h\mathbf u_t+\mathbf b_h)+\mathbf b_o.
$$

The final output is linear, so the generator is not artificially bounded. The
paper does not unambiguously specify the hidden dense activation; LEEM records
its explicit implementation choice, Leaky-ReLU with slope $0.2$, in every saved
configuration.

### 3.2 Discriminator and LEEM mask adaptation

Let $\mathbf m_t\in\{0,1\}^{K}$ denote target availability. The original paper
did not require missing spatial target dimensions. LEEM gives the discriminator

$$
[\mathbf m_t\odot\mathbf y_t,\mathbf m_t,\mathbf x_t].
$$

The same mask is applied to real and generated targets. Supplying the mask
prevents an unavailable look-ahead station, stored as zero, from being mistaken
for an observed zero-metre error. After the discriminator LSTM, the raw condition
is skip-connected to its dense head and the result is a logit $a_t$:

$$
a_t=D(\mathbf y_{0:t},\mathbf x_{0:t},\mathbf m_{0:t}),
\qquad p_t=\sigma(a_t).
$$

Only active frames containing at least one observed target enter the loss.
Padding therefore cannot affect the objective.

### 3.3 Non-saturating conditional GAN objective

Let $\mathcal A$ be the set of eligible frames in a batch. The discriminator
minimizes binary cross-entropy for real and generated sequences:

$$
\mathcal L_D=-\frac{1}{|\mathcal A|}
\sum_{t\in\mathcal A}
\left[\log\sigma(a_t^{\mathrm{real}})
+\log(1-\sigma(a_t^{\mathrm{fake}}))\right].
$$

The generator uses the paper's non-saturating loss:

$$
\mathcal L_G=-\frac{1}{|\mathcal A|}
\sum_{t\in\mathcal A}\log\sigma(a_t^{\mathrm{fake}}).
$$

The losses are means over active time steps so long sequences do not receive
larger weight merely because they contain more frames.

### 3.4 Validation selection by Energy Score

RC-GAN is an implicit generative model; it does not provide a tractable
normalized likelihood. Initialization restarts are therefore ranked on the
validation split with the same sample-based proper score used in the three-model
comparison. For observed station subset $\mathcal O_t$ and $S$ draws,

$$
\operatorname{ES}_t=
\frac{1}{S}\sum_{s=1}^{S}
\lVert\widehat{\mathbf y}^{(s)}_{t,\mathcal O_t}
-\mathbf y_{t,\mathcal O_t}\rVert_2
-\frac{1}{2S^2}\sum_{r=1}^{S}\sum_{s=1}^{S}
\lVert\widehat{\mathbf y}^{(r)}_{t,\mathcal O_t}
-\widehat{\mathbf y}^{(s)}_{t,\mathcal O_t}\rVert_2.
$$

LEEM divides each frame score by $\sqrt{|\mathcal O_t|}$ before averaging so
frames with different visible ranges remain comparable. Selection is performed
in physical metres after inverse standardization. The held-out test files are
not opened until the restart is frozen.

## 4. Implementation

### 4.1 Paper fidelity and explicit adaptations

| Choice | Paper basis | LEEM implementation |
|---|---|---|
| Noise input | Gaussian vector at every frame | $\mathbf z_t\sim\mathcal N(0,I)$, dimension 32 in prototype |
| Noise recurrent path | Separate one-layer LSTM | Preserved |
| Context recurrent path | Deep LSTM; final conference model uses two layers | Two layers in prototype |
| Condition skip | Generator and discriminator | Preserved |
| Output | Linear | Preserved |
| Discriminator dropout | 5% | Preserved |
| Optimizer | Adam, $\beta_1=0.5$, $\beta_2=0.999$ | Preserved |
| Learning rate | $10^{-5}$ | Preserved in prototype |
| Training epochs | Four | Preserved in prototype |
| LSTM initialization | Truncated normal, standard deviation 0.1 | Preserved |
| Dense initialization | Xavier | Preserved |
| Variable length | Sequence loss averaged over time | Explicit padding mask |
| Missing stations | Not described | Target values and loss are masked; mask is discriminator input |
| Gradient clipping | Not reported | Norm 1.0 stability safeguard, persisted as an adaptation |
| Dense hidden activation | Not explicit | Leaky-ReLU, slope 0.2, persisted as an implementation choice |

The paper used batch size one. The LEEM model default remains one. The Phase 7.1
pilot and provisional prototype configuration used batch size four as a
transparent runtime compromise, but Phases 7.2 and 7.3 restore batch size one.
The final prototype configuration must be frozen from the Phase 7.3 decision
before use.

The smoke configuration is intentionally tiny: one epoch, small hidden layers,
one restart, and a larger learning rate. It is a software gate only and must not
be used as a thesis result.

### 4.2 Shared data contract

The model consumes standardized arrays:

- conditions `[B,T,6]`;
- errors `[B,T,21]`;
- target availability `[B,T,21]`;
- lengths `[B]`.

It returns samples `[S,B,T,21]`. Padding is exactly zero. Generated values at an
active but unobserved station are retained: the mask controls training and
evaluation availability, not whether the model can produce a full profile.

### 4.3 Determinism and persistence

Training shuffles complete sequences with an epoch-specific seed. Sampling uses
a local PyTorch random generator, so repeated calls with the same fitted model,
inputs, and seed are identical. The thesis configurations use CPU execution for
the strongest reproducibility; CUDA is supported when explicitly selected but
hardware/library differences must be recorded.

The model is stored as `rcgan_model.npz`, not a pickle-based `.pt` file. The
archive contains both network state dictionaries as named NumPy arrays, complete
configuration, feature order, look-ahead grid, training history, schema version,
and training sequence count. Loading uses `allow_pickle=False`.

### 4.4 Installation and commands

Install PyTorch and the common plotting evaluator:

```powershell
python -m pip install -e ".[evaluation,rcgan]"
```

Run the fast software gate first:

```powershell
python scripts/run_rcgan_experiment.py `
  --config configs/rcgan_experiment_smoke.json `
  --output outputs/experiments/rcgan_smoke
```

The smoke manifest proves that the software path works, but it does not establish
that the generator uses its noise. Generate and run the intermediate
conditional-Gaussian stability pilot next:

```powershell
generate-lane-error-data `
  --config configs/synthetic_rcgan_pilot.json `
  --output outputs/synthetic_rcgan_pilot

python scripts/run_rcgan_experiment.py `
  --config configs/rcgan_experiment_pilot.json `
  --output outputs/experiments/rcgan_pilot
```

The Phase 7.1 pilot uses the full paper-sized architecture on 128/32/32 sequences
and compares learning rates $10^{-5}$, $10^{-4}$, and $3\times10^{-4}$ using the
validation split only. Its selected candidate passed the original diversity
threshold but remained severely under-dispersed on the test set, while the two
higher rates showed persistent clipping and discriminator domination. Therefore,
do not run the prototype from the Phase 7.1 result.

Reuse the same data for the Phase 7.2 stabilization gate:

```powershell
python scripts/run_rcgan_experiment.py `
  --config configs/rcgan_experiment_pilot_v2.json `
  --output results/synthetic/rcgan_pilot_v2
```

Phase 7.2 compared $10^{-5}$, $3\times10^{-5}$, and $5\times10^{-5}$ at the
paper's batch size one. All three candidates failed: the lower rates remained
conditionally under-dispersed, while the highest rate showed severe generator
clipping and excessive tails. The runner persisted `status=stability_failed`
without opening the test split.

Phase 7.3 is the final bounded, architecture-preserving attempt. It fixes the
discriminator rate at $10^{-5}$ and tests generator rates of
$3\times10^{-5}$ and $5\times10^{-5}$, adds noise-path gradient diagnostics,
checks the worst late epoch, and rejects both missing and explosive tails:

```powershell
python scripts/run_rcgan_experiment.py `
  --config configs/rcgan_experiment_pilot_v3.json `
  --output results/synthetic/rcgan_pilot_v3
```

Only if Phase 7.3 passes should its selected optimizer rates and batch size be
frozen in `rcgan_experiment_prototype.json`. The currently committed prototype
settings are provisional. Then run the prototype:

```powershell
python scripts/run_rcgan_experiment.py `
  --config configs/rcgan_experiment_prototype.json `
  --output outputs/experiments/rcgan_prototype
```

The prototype fits two deterministic restarts for each of three scenarios and
can take hours. Runtime depends strongly on CPU, thread libraries, and sequence
lengths. Keep the terminal open and do not use the smoke result in model claims.

### 4.5 Epoch-wise stability diagnostics

Every epoch records generator/discriminator losses, pre-clipping gradient norms,
gradient-clipping fractions, discriminator real/fake probabilities, and the
conditional diversity ratio

$$
r_{\mathrm{div}}=
\frac{\operatorname{mean}_{b,t,k}
\operatorname{Std}_{s}(\widehat y^{(s)}_{btk}\mid\mathbf x_{bt})}
{\operatorname{Std}(y_{btk})}.
$$

The numerator varies latent noise while holding conditions fixed. The denominator
only makes the diagnostic scale-free. This ratio is an engineering collapse
indicator, not a calibration metric and not part of the three-model leaderboard.
Phase 7.3 rejects candidates below the predeclared threshold
$r_{\mathrm{div}}=0.10$ and applies the additional optimization, discriminator,
coverage, and two-sided tail checks described in [Phase 7.3 RC-GAN
stabilization](phase7_3_rcgan_asymmetric_optimizer.md). The smoke gate keeps all
rejections disabled because its tiny data and one epoch are not scientific.

## 5. Current status

Phase 7 implements:

- the paper-based generator and discriminator;
- masked variable-length adversarial training;
- deterministic full-profile sampling;
- safe model persistence and exact round-trip tests;
- validation-only stability filtering followed by physical-unit normalized
  Energy Score selection;
- epoch-wise loss, discriminator-probability, gradient, and conditional-diversity
  diagnostics;
- held-out common evaluation, plots, provenance, and synthetic oracle mean
  diagnostics;
- smoke and prototype configurations.

The automated smoke runner and complete unit suite pass. Phase 7.1 also ran to
completion, but its result was not statistically ready for the prototype. Phase
7.2 rejected every candidate without opening the test split. Phase 7.3 is now
the final bounded gate. Passing tests establish software correctness, not
statistical superiority or publication evidence.

## 6. Key takeaway

Phase 7 completes the planned three-model implementation without changing the
frozen LEEM data or evaluation contract. The RC-GAN is faithful to the selected
paper where the paper is explicit, and every necessary LEEM adaptation is visible
in configuration, code, and documentation. The next scientific gate is Phase
7.3—not the full prototype. The prototype follows only if at least one
predeclared Phase 7.3 candidate passes. If none passes, RC-GAN remains the third
thesis model as a controlled negative result and synthetic tuning stops.
