# Autoregressive Input-Output Hidden Markov Model

## 1. Purpose and relationship to the reference paper

The second LEEM model is an autoregressive input-output hidden Markov model
(AIOHMM). It represents a lane-estimation error sequence with a discrete latent
regime, condition-dependent regime transitions, autoregressive temporal memory,
and a state-specific spatial error distribution.

The implementation follows the model family in Zec et al., *Statistical Sensor
Modelling for Autonomous Driving Using Autoregressive Input-Output HMMs*
([paper PDF](https://schlieplab.org/Static/Publications/2018-ITSC-AutoregressiveHMM.pdf)).
It is an adaptation, not an exact reproduction. The paper demonstrates a scalar
sensor output. LEEM must generate a 21-station lateral-error profile, so it adds a
full state-specific spatial covariance and station-specific diagonal
autoregression while preserving the paper's input-dependent hidden-state
transition mechanism.

The synthetic experiments are implementation and capability tests. They do not
show that BMW lane-estimation errors have hidden regimes or follow this model.

## 2. Variables and parameterization

For one sequence, let:

- $\mathbf x_t\in\mathbb R^F$ be the standardized condition vector;
- $\mathbf y_t\in\mathbb R^K$ be the standardized signed error profile;
- $z_t\in\{1,\ldots,N\}$ be the latent state;
- $\mathbf d_t=[1,\mathbf x_t^\mathsf T]^\mathsf T$ be the transition and
  emission design vector.

LEEM currently uses $F=6$, $K=21$, and validation-selected $N$. Parameters are:

| Parameter | Shape | Meaning |
|---|---:|---|
| $\boldsymbol\pi$ | $[N]$ | Initial-state probabilities |
| $W$ | $[N,N,F+1]$ | Input-dependent transition weights |
| $B$ | $[N,F+1,K]$ | State-specific conditional-mean coefficients |
| $\mathbf a$ | $[N,K]$ | Station-wise AR(1) coefficients |
| $\Sigma$ | $[N,K,K]$ | State-specific spatial innovation covariances |

The initial state follows

$$
p(z_0=j)=\pi_j.
$$

For $t\geq1$, the transition probability from state $i$ to $j$ is

$$
p(z_t=j\mid z_{t-1}=i,\mathbf x_t)
=
\frac{\exp(W_{ij}^{\mathsf T}\mathbf d_t)}
{\sum_{r=1}^{N}\exp(W_{ir}^{\mathsf T}\mathbf d_t)}.
$$

The multivariate autoregressive emission is

$$
\mathbf y_t\mid z_t=k,\mathbf x_t,\mathbf y_{t-1}
\sim
\mathcal N(\boldsymbol\mu_{t,k},\Sigma_k),
$$

$$
\boldsymbol\mu_{t,k}
=B_k^{\mathsf T}\mathbf d_t
+\mathbf a_k\odot\mathbf y_{t-1}^{\mathrm{available}}.
$$

The AR matrix is deliberately diagonal: a station depends on its own previous
value, while simultaneous dependence between stations is represented by the full
$\Sigma_k$. This controls the parameter count for the initial thesis comparison.
A full $K\times K$ lag matrix would add $441$ lag coefficients per state and is
not justified before the available BMW sample size and missingness are known.

## 3. Missing targets and variable-length sequences

Likelihood calculations use the common `valid_mask`. If only station subset
$\mathcal O_t$ is observed at time $t$, the emission contribution is the
marginal Gaussian

$$
p(\mathbf y_{t,\mathcal O_t}\mid z_t=k)
=
\mathcal N(
\mathbf y_{t,\mathcal O_t};
\boldsymbol\mu_{t,k,\mathcal O_t},
\Sigma_{k,\mathcal O_t,\mathcal O_t}).
$$

Unobserved current stations therefore contribute no artificial zero-valued
targets. If the preceding value of a station is unavailable, its AR input is set
to zero in standardized space, which is the training mean for that station. The
same rule is used at the first frame. This is an explicit approximation: the
missing previous value is not integrated out as an additional latent variable.

Padding frames are excluded using sequence lengths. Sampling generates a full
profile on every active frame; the real-data mask controls evaluation
availability rather than suppressing generated stations.

## 4. Forward-backward inference

For state $k$, define the observed-dimension emission log density
$\ell_t(k)$. The forward recursion is

$$
\alpha_0(k)=\pi_k\exp(\ell_0(k)),
$$

$$
\alpha_t(j)=
\exp(\ell_t(j))
\sum_i\alpha_{t-1}(i)A_t(i,j;\mathbf x_t).
$$

The implementation evaluates this recursion and the corresponding backward
recursion in the log domain with per-frame normalization. It returns:

$$
\gamma_t(k)=p(z_t=k\mid\mathbf x_{0:T},\mathbf y_{0:T}),
$$

$$
\xi_t(i,j)=p(z_t=i,z_{t+1}=j\mid\mathbf x_{0:T},\mathbf y_{0:T}).
$$

An exact state-enumeration unit test verifies forward-backward probabilities on a
small HMM. Factorizations of repeated observed-station covariance submatrices are
cached during a likelihood pass.

## 5. Generalized EM fitting

One deterministic run uses generalized expectation-maximization.

### E-step

Forward-backward inference computes $\gamma$ and $\xi$ for every complete
training sequence. Frames are never shuffled independently.

### Initial probabilities

The initial distribution is updated from posterior counts with positive additive
smoothing:

$$
\pi_k
=
\frac{\epsilon+\sum_i\gamma_{i,0}(k)}
{N\epsilon+\sum_r\sum_i\gamma_{i,0}(r)}.
$$

### Emission regression

For every state and station, weighted ridge regression estimates the intercept,
condition effects, and the station's lag coefficient. Only frames with a valid
current target at that station are included, and weights are the state posterior
$\gamma_t(k)$. The intercept is not penalized. AR coefficients are clipped to
$[-a_{\max},a_{\max}]$ with $a_{\max}<1$ to limit unstable recursive samples.

### Spatial covariance

State-weighted residual cross-products use every available station pair. A pair
with insufficient effective observations is left at zero before regularization.
The raw matrix $C_k$ is shrunk toward its diagonal,

$$
C_k^{\mathrm{shrunk}}
=(1-\lambda)C_k+\lambda\operatorname{diag}(C_k),
$$

then symmetrized and projected to a configured minimum eigenvalue. This makes
every saved state covariance positive definite while retaining within-frame
spatial correlation where supported by data.

### Input-dependent transitions

There is no closed-form update for the multinomial-logistic transition weights.
The code maximizes the expected transition log likelihood

$$
\sum_{i,t,r,s}\xi_{i,t}(r,s)
\log A_{i,t}(r,s;\mathbf x_{i,t+1})
-\frac{\lambda_W}{2}\lVert W_{:,:,1:}\rVert_2^2
$$

with a fixed number of deterministic Adam steps. Intercepts are not penalized.
Destination logits are centred after every step because adding the same vector
to every destination logit leaves a softmax unchanged.

Because transition optimization is numerical and covariance estimation with
missing pairs is approximate, an iteration is not guaranteed to increase the
observed-data likelihood. The implementation retains the best observed training
likelihood state and reports any decreases. This is why the procedure is
described as generalized EM rather than an exact closed-form EM algorithm.

## 6. Initialization, restarts, and state selection

Initial assignments are based on quantiles of each frame's observed standardized
profile RMS. Small seeded jitter makes deterministic restarts distinct without
using validation or test information. Smoothed hard assignments initialize
emissions and transitions.

The candidate grid combines:

- hidden-state counts;
- initialization seeds.

Every candidate is fitted on training data and ranked by validation negative log
likelihood per observed standardized target. Ties prefer fewer states and then
the lower seed. Test data are first loaded after this choice is frozen. The
selected deterministic configuration is fitted again on training only, keeping
the same split rule used for the Gaussian baseline.

Hidden-state numbers are labels, not physical classes. Permuting all
state-specific parameters produces the same model. State interpretations must be
based on fitted parameter/posterior summaries and cannot be assumed from a state
index.

## 7. Recursive sampling

For each requested stochastic sample and sequence:

1. draw $z_0$ from $\boldsymbol\pi$;
2. draw $\mathbf y_0$ from its state-conditional multivariate Gaussian;
3. at each later frame, draw $z_t$ from the condition-dependent transition row;
4. compute the state-specific conditional mean using the generated
   $\mathbf y_{t-1}$;
5. draw the full spatial innovation through the Cholesky factor of $\Sigma_{z_t}$.

The supplied seed completely determines the output. Generated padding is exactly
zero. Sampling is free-running: it uses previous generated values, whereas
training likelihood conditions on previous observed values when available.

## 8. Evaluation and diagnostics

Primary comparison with Gaussian and RC-GAN uses the common physical-unit sample
metrics: predictive mean errors, CRPS, energy score, calibration, marginal and
first-difference distances, tail errors, temporal residual correlation, and
spatial residual correlation.

AIOHMM additionally reports secondary diagnostics:

- validation and test NLL per observed standardized value;
- posterior state occupancy and entropy;
- average self-transition probability;
- variability of transition probabilities over test conditions;
- minimum/maximum AR coefficients;
- minimum covariance eigenvalue.

For synthetic data only, normalized mutual information compares the maximum
posterior state with the generator's oracle latent state. Oracle state labels are
never used for training or selection. This diagnostic is unavailable and
inappropriate for BMW data.

## 9. Persistence and reproducibility

The compressed NPZ artifact stores the complete configuration, feature order,
look-ahead grid, all fitted parameters, state occupancies, likelihood history,
training sequence count, model name, and schema version. Loading uses
`allow_pickle=False`, validates dimensions and numerical constraints, and
reconstructs covariance Cholesky factors.

Experiment manifests record source and artifact SHA-256 checksums, every
candidate result, seeds, split provenance, metrics, and warnings. Generated
experiments and BMW data must not be committed to the public repository.

## 10. Assumptions and limitations

- The multivariate extension is LEEM-specific and is not an exact reproduction
  of the scalar reference-paper experiment.
- Diagonal AR terms cannot represent cross-station lag effects.
- Covariance is constant within a state and cannot vary smoothly with conditions.
- Gaussian state emissions approximate heavy tails only through a finite mixture.
- The model has no explicit duration distribution; persistence is geometric after
  conditioning on inputs.
- The initial-state distribution does not depend on the first condition.
- The missingness mechanism itself is not modelled, and a missing previous target
  is replaced by the standardized mean in the likelihood regression.
- Teacher-forced likelihood and free-running sampling can behave differently,
  especially when AR coefficients approach their stability bound.
- Small synthetic smoke splits are suitable for code checks, not precise state
  count or hyperparameter conclusions.
- A good synthetic latent-state match does not establish physically meaningful
  regimes in BMW data.

These limitations are part of the model comparison. In particular, remaining
spatial-correlation and tail errors provide concrete questions for the RC-GAN
rather than reasons to hide negative results.

## 11. Commands

Run focused AIOHMM tests:

```powershell
python -m unittest tests.test_aiohmm_inference `
  tests.test_aiohmm_model tests.test_aiohmm_experiment -v
```

Run the three-scenario smoke experiment:

```powershell
python scripts/run_aiohmm_experiment.py `
  --config configs/aiohmm_experiment_smoke.json `
  --output outputs/experiments/aiohmm_smoke
```

Only after the smoke gate passes, run the prototype configuration:

```powershell
python scripts/run_aiohmm_experiment.py `
  --config configs/aiohmm_experiment_prototype.json `
  --output outputs/experiments/aiohmm_prototype
```
