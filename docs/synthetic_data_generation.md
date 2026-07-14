# Synthetic Lane-Estimation Error Data Generation Methodology

## 1. Purpose and scientific role

The synthetic pipeline provides a controlled environment for implementing and verifying the Gaussian, AIOHMM, and RC-GAN model families before BMW data are requested. Its objectives are to verify:

- the mathematical lane-error definition;
- variable-length sequence handling;
- feature and target tensor compatibility;
- spatially correlated multivariate sampling;
- temporal and latent-state learning capability;
- mask-aware training and evaluation;
- deterministic experiment reproduction;
- geometrically consistent injection into the lateral trajectory planner.

It does not estimate the distribution of real perception error. Its parameters are deliberately explicit and are selected to produce distinguishable, numerically plausible test processes rather than to imitate confidential BMW signals.

The three scenarios must not be aggregated into a single synthetic leaderboard. Each scenario answers a capability question:

1. Can a model recover a conditional Gaussian process when its assumptions are correct?
2. Can a model recover regime persistence and autoregression?
3. Can a model represent nonlinear condition interactions, long memory, heavy tails, and rare persistent excursions?

## 2. Notation and coordinate convention

For sequence $i$, time index $t$, and look-ahead station $s_k$:

- $\mathbf p^*_{t,k}\in\mathbb R^2$: reference-path point;
- $\hat{\mathbf p}_{t,k}\in\mathbb R^2$: perceived/estimated path point;
- $\theta^*_{t,k}$: reference heading;
- $\mathbf n^*_{t,k}=[-\sin\theta^*_{t,k},\cos\theta^*_{t,k}]^\mathsf T$: left-pointing reference normal;
- $e_{t,k}$: signed lateral estimation error;
- $\mathbf X_t\in\mathbb R^6$: condition vector;
- $\mathbf Y_t=[e_{t,0},\ldots,e_{t,K-1}]^\mathsf T$: error profile.

The error is

$$
e_{t,k}=
(\hat{\mathbf p}_{t,k}-\mathbf p^*_{t,k})^\mathsf T
\mathbf n^*_{t,k}.
$$

Positive error lies to the left of the reference-path direction. The development grid is

$$
s_k=5k\ \text{m},\qquad k=0,\ldots,20.
$$

Thus $K=21$ and the maximum development look-ahead is 100 m. The grid is configuration-controlled rather than hard-coded in the model implementations.

## 3. Sequence and split definition

The development sampling frequency is 10 Hz. Prototype sequence lengths are sampled uniformly from 50 through 200 frames, corresponding to 5-20 seconds. Every generated sequence is statistically independent at initialization and has its own deterministic seed.

For each DGP scenario, the prototype configuration creates:

| Split | Sequences | Permitted use |
|---|---:|---|
| Training | 2,000 | Parameter estimation and training |
| Validation | 400 | Model selection and early stopping |
| Test | 400 | Final synthetic capability evaluation |

The split identifier is included in seed derivation, so the numerical streams are disjoint even for equal sequence indices. Synthetic splitting does not imitate real-drive splitting; BMW data must later be split by complete drives, routes, vehicles, or scenarios.

## 4. Reference-scene generation

### 4.1 Ego speed

Initial speed is drawn uniformly within the configured range. Acceleration follows a clipped first-order process:

$$
a_t=\operatorname{clip}(0.94a_{t-1}+\eta^a_t,-2.5,2.0),
\qquad
\eta^a_t\sim\mathcal N(0,0.12^2),
$$

and speed is integrated with $\Delta t=0.1$ s:

$$
v_t=\operatorname{clip}(v_{t-1}+a_t\Delta t,v_{\min},v_{\max}).
$$

Acceleration is an internal generator variable and is not automatically supplied to the statistical models.

### 4.2 Curvature profile

The reference curvature at time $t$ is a linear function of arc length:

$$
\kappa_t(s)=
\operatorname{clip}
\left(
\kappa_{0,t}+g_t(s-\bar s),
-\kappa_{\max},\kappa_{\max}
\right).
$$

The center curvature and spatial gradient evolve smoothly:

$$
\kappa_{0,t}=\operatorname{clip}
(0.992\kappa_{0,t-1}+\eta^\kappa_t,-\kappa_{\max},\kappa_{\max}),
$$

$$
g_t=\operatorname{clip}
(0.985g_{t-1}+\eta^g_t,-g_{\max},g_{\max}).
$$

The default innovation standard deviations are $4.5\times10^{-4}\ \mathrm{m}^{-1}$ and $8\times10^{-6}\ \mathrm{m}^{-2}$, respectively. The two model inputs derived from this profile are

$$
\bar\kappa_t=\frac{1}{K}\sum_k\kappa_t(s_k),
\qquad
\Delta\kappa_t=\max_k\kappa_t(s_k)-\min_k\kappa_t(s_k).
$$

### 4.3 Reference-path integration

Heading is integrated from curvature by the trapezoidal rule:

$$
\theta_k=\theta_{k-1}
+\frac{\kappa_{k-1}+\kappa_k}{2}(s_k-s_{k-1}).
$$

Position is then integrated using the midpoint heading:

$$
\mathbf p_k=\mathbf p_{k-1}+
(s_k-s_{k-1})
\begin{bmatrix}
\cos((\theta_{k-1}+\theta_k)/2)\\
\sin((\theta_{k-1}+\theta_k)/2)
\end{bmatrix}.
$$

Every frame is represented in an ego-local frame beginning at $\mathbf p_0=(0,0)$ and $\theta_0=0$. This is a reference-geometry generator, not a globally continuous vehicle localization simulation.

### 4.4 Lane width and quality conditions

Lane width follows a slowly varying bounded process around common road-lane widths. Marking and environmental quality are separate bounded switching processes. Each quality process alternates between favourable and degraded targets, with persistence and noisy relaxation:

$$
q_t=\operatorname{clip}(0.92q_{t-1}+0.08q_{\text{target},t}+\eta^q_t,0.02,1).
$$

These values are generic quality indices. They are not claimed to correspond to an existing BMW signal. The later data-mapping stage must define a documented construction from available measurements or labels.

### 4.5 Fixed condition vector

All three model families receive

$$
\mathbf X_t=
[v_t,\bar\kappa_t,\Delta\kappa_t,w_{\mathrm{lane},t},
q_{\mathrm{mark},t},q_{\mathrm{env},t}]^\mathsf T.
$$

Oracle state variables, scenario labels, future errors, conditional means, and validity masks are never model inputs.

## 5. Spatial dependence

The base spatial correlation between stations is exponential:

$$
R_{k\ell}=\exp\left(-\frac{|s_k-s_\ell|}{\lambda}\right).
$$

If $\boldsymbol\sigma$ is the station-wise standard deviation, the covariance is

$$
\boldsymbol\Sigma=
\operatorname{diag}(\boldsymbol\sigma)
\mathbf R
\operatorname{diag}(\boldsymbol\sigma).
$$

A small diagonal numerical jitter is added before Cholesky factorization. Station variance increases with look-ahead distance because far-path estimates are expected to be a harder modelling case; this is a controlled assumption, not a measured real-data conclusion.

## 6. Scenario 1: conditional multivariate Gaussian

The first process matches the Gaussian baseline assumptions:

$$
\mathbf Y_t\mid\mathbf X_t
\sim\mathcal N(\boldsymbol\mu(\mathbf X_t),\boldsymbol\Sigma),
$$

independently over time conditional on $\mathbf X_t$. The mean is linear in normalized conditions, with station-dependent coefficients. Let $r_k=s_k/s_{K-1}$ and define normalized conditions

$$
\tilde v=(v-20)/15,\quad
\tilde\kappa=\bar\kappa/0.01,\quad
\widetilde{\Delta\kappa}=\Delta\kappa/0.02,
$$

$$
\tilde w=(w-3.6)/0.4,\quad
b_m=1-q_{\mathrm{mark}},\quad
b_e=1-q_{\mathrm{env}}.
$$

The implemented mean is

$$
\mu_{t,k}=
0.006r_k
+0.016\tilde v_t r_k
+0.065\tilde\kappa_t r_k^{1.35}
+0.025\widetilde{\Delta\kappa}_t r_k
+0.012\tilde w_t r_k
+0.055b_{m,t}r_k^{1.6}
+0.035b_{e,t}r_k^{1.4}.
$$

The station standard deviation is

$$
\sigma_k=0.012+0.095r_k^{1.35}\ \text{m},
$$

and $\lambda=25$ m. This scenario checks parameter recovery, likelihood, covariance estimation, sampling, and the absence of unexplained residual temporal correlation.

## 7. Scenario 2: latent autoregressive regimes

The second process has three latent regimes:

1. good;
2. biased;
3. degraded.

The base transition matrix is

$$
\mathbf A_0=
\begin{bmatrix}
0.965&0.028&0.007\\
0.075&0.885&0.040\\
0.035&0.125&0.840
\end{bmatrix}.
$$

At every time step, normalized speed, curvature, curvature variation, marking quality, and environmental quality form a non-negative risk value $r_t$. Transition logits are

$$
\ell_{ij,t}=\log A_{0,ij}+r_t[-1.0,0.25,1.15]_j,
$$

followed by a softmax. Increasing risk therefore shifts transition probability toward the degraded state while retaining dependence on the previous state.

The emission process is

$$
\mathbf Y_t=
\phi_{Z_t}\mathbf Y_{t-1}
+(1-\phi_{Z_t})
[\boldsymbol\mu(\mathbf X_t)+\mathbf b_{Z_t}]
+\boldsymbol\epsilon_{t,Z_t},
$$

where

$$
(\phi_0,\phi_1,\phi_2)=(0.45,0.70,0.88).
$$

State-specific bias profiles, innovation scales, and spatial correlation lengths create persistent but distinguishable regimes. The biased-state sign depends on the observed signed curvature condition rather than an unobserved per-sequence random effect. The oracle state is stored only to evaluate state recovery and is never exposed during model fitting.

## 8. Scenario 3: nonlinear recurrent heavy-tailed process

The third process introduces effects outside the first two model assumptions.

A three-dimensional latent memory evolves as

$$
\mathbf h_t=0.92\mathbf h_{t-1}
+0.08\tanh(\mathbf W\tilde{\mathbf X}_t)
+\boldsymbol\eta^h_t.
$$

The target profile contains nonlinear curvature/distance effects, speed-curvature interaction, multiplicative marking-environment degradation, and memory-dependent profiles. Error follows

$$
\mathbf Y_t=
\phi_t\mathbf Y_{t-1}
+(1-\phi_t)\mathbf m(\mathbf X_t,\mathbf h_t)
+\mathbf b^{\mathrm{burst}}_t
+\boldsymbol\epsilon_t,
$$

where $\phi_t$ is a bounded nonlinear function of condition risk.

Innovations use a spatially correlated multivariate Student-like construction with seven degrees of freedom. This retains heavy tails while avoiding implausibly frequent multi-metre numerical outliers in a large synthetic dataset. A small centered quadratic transformation creates conditional skewness. Rare bursts have:

- condition-dependent onset probability;
- 0.93 one-step persistence probability;
- slowly decaying amplitude;
- random sign, center, and spatial width.

The burst flag is stored as an oracle binary state. This scenario checks sample diversity, tail representation, long-memory behaviour, rare-event duration, and mode-collapse diagnostics. It is not designed to guarantee RC-GAN superiority.

## 9. Validity masks and missing look-ahead

Synthetic visible range is

$$
r_{\mathrm{valid},t}=\min\left(
s_{K-1},
40+80(0.6q_{\mathrm{mark},t}+0.4q_{\mathrm{env},t})
\right)\ \text{m}.
$$

A target station is valid when $s_k\le r_{\mathrm{valid},t}$. The formula makes the complete 100 m profile observable in favourable conditions while shortening the range during degraded intervals. Invalid error entries are serialized as zero only for numerical storage and are always marked `False` in `valid_mask`. Losses and metrics must use the mask. Treating stored zeros as observations would bias means, variances, and far-range performance.

## 10. Reproducibility and provenance

Each sequence seed is derived from four stable integers:

$$
S_i=f(S_{\mathrm{master}},S_{\mathrm{scenario}},S_{\mathrm{split}},i).
$$

Consequences:

- generation order does not change a sequence;
- train/validation/test streams are disjoint;
- one failed or suspicious sequence can be regenerated independently;
- increasing a split size preserves all existing lower-index sequences.

Every output manifest records:

- complete configuration;
- schema and generator versions;
- feature names and target definition;
- SHA-256 file checksums;
- sequence counts and maximum lengths;
- valid fractions;
- condition ranges;
- error mean, standard deviation, and absolute 95th/99th/max quantiles;
- latent-state occupancy.

## 11. Required validation before model training

Generation is accepted only when:

1. configuration and physical bounds validate;
2. all tensor shapes follow the schema;
3. valid values are finite;
4. padding is zero and fully masked;
5. quality features remain in $[0,1]$;
6. errors remain below the configured numerical plausibility guard;
7. signed normal error is recovered after path perturbation;
8. repeated generation is bitwise identical;
9. split seeds are distinct;
10. scenario diagnostics reveal the intended Gaussian, regime, and heavy-tail properties.

The plausibility guard is a software protection, not an empirical safety threshold. Its final value must be reconsidered using real-data units and ranges.

## 12. Planner injection

For a sampled error vector $\mathbf Y_t$:

1. construct normals from the ideal reference heading;
2. shift reference points by $e_{t,k}\mathbf n^*_{t,k}$;
3. fit or resample a smooth perceived path;
4. derive perceived heading and curvature from that same path;
5. calculate perceived lateral boundaries consistently;
6. pass the corrupted reference geometry to the lateral QP planner;
7. evaluate the planned trajectory relative to the ideal road.

Independent noise must not be added separately to lateral offset, heading, and curvature because it can create mutually inconsistent geometry.

## 13. Mapping contract for later BMW data

After all three models pass synthetic tests, the BMW request must provide enough information to construct:

- stable drive and sequence IDs;
- synchronized estimated and reference path geometries;
- coordinate transforms and calibration versions;
- sampling timestamps and latency information;
- reference-system accuracy indicators;
- ego speed;
- reference curvature and lane width or geometry from which they can be derived;
- lane-marking quality/visibility;
- environmental visibility/illumination;
- valid range and availability flags;
- planner horizon, path grid, initial state, constraints, and outputs for integration.

The 10 Hz and 0-100 m development settings are requested if available, but the adapters may resample real data to a different fixed grid/frequency after documenting the reason. The statistical error definition must not change during mapping.

## 14. Threats to validity

- Synthetic feature-error relationships are authored assumptions.
- Smooth AR processes do not reproduce all real driving transitions.
- Reference geometry is ego-local per frame and does not simulate localization drift.
- Missing detections are represented by range masks, not a full detection/non-detection model.
- Marking and environment quality are generic indices without a current BMW signal mapping.
- The maximum look-ahead may not match the final planner configuration.
- Heavy-tail and burst parameters are capability tests, not measured event frequencies.
- A model performing well on its matching synthetic DGP may fail on BMW data.
- Synthetic results cannot support claims of real-world calibration, robustness, or safety benefit.

These limitations must be stated whenever synthetic results appear in the thesis or paper.
