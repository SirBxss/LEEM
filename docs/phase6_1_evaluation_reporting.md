# Phase 6.1: Evaluation Reporting and Result Comparison

## Purpose

Phase 6.1 closes two reporting gaps found after the Gaussian and AIOHMM
synthetic prototype runs:

1. central prediction intervals were estimated from only 64 generated draws;
2. the two experiment trees had no automatic, comparability-checked summary.

This phase does not retrain either model and does not change the scientific
conclusions of the prototype experiment.

## Finite-ensemble interval reporting

For a nominal central interval with coverage \(c\), the evaluator uses lower
and upper probabilities

\[
p_L=\frac{1-c}{2}, \qquad p_U=1-p_L.
\]

The existing implementation uses NumPy's linear sample quantiles. With
\(n\) predictive draws and a continuous uniform reference distribution, the
expected probability-scale position of a linearly interpolated quantile is

\[
\mathbb E[Q(p)] = \frac{(n-1)p+1}{n+1}.
\]

Therefore, the finite-ensemble reference coverage is

\[
c_{\mathrm{linear,ref}}
=\frac{(n-1)(p_U-p_L)}{n+1}
=\frac{(n-1)c}{n+1}.
\]

For \(n=64\), the reference values are approximately:

| Nominal coverage | Linear uniform-reference coverage |
|---:|---:|
| 0.50 | 0.4846 |
| 0.90 | 0.8723 |
| 0.95 | 0.9208 |

This explains why empirical coverages near 0.48, 0.87, and 0.92 are not, by
themselves, evidence of model miscalibration. The reference is diagnostic,
not a universal correction: interpolation occurs in measurement space, so
the exact expectation depends on the predictive distribution outside the
uniform reference case.

Each newly produced `evaluation.json` stores:

- the nominal and empirical coverage;
- the linear finite-ensemble reference coverage;
- empirical coverage minus that reference;
- the nearest conservative central order-statistic interval and its
  attainable reference coverage;
- an explicit interpretation warning.

Existing evaluation files can be upgraded without samples or model fitting:

```bash
python scripts/upgrade_evaluation_results.py \
  --root results/synthetic \
  --write
```

The command is read-only unless `--write` is supplied.

For publication-level calibration claims, use a substantially larger common
predictive sample count for every model or pre-register a rank-based interval
definition. Do not compare raw 64-draw empirical coverage directly with the
nominal level.

## Automatic Gaussian-versus-AIOHMM comparison

The comparison command reads only persisted evaluation artifacts:

```bash
python scripts/compare_experiments.py \
  --baseline results/synthetic/gaussian_prototype \
  --candidate results/synthetic/aiohmm_prototype \
  --output results/synthetic/gaussian_vs_aiohmm
```

Before calculating any difference, it verifies that both models use:

- exactly the same scenario set;
- the same sequence count and observed-value count;
- the same look-ahead station grid;
- byte-identical `evaluation_reference.json` files for each scenario.

It writes deterministic JSON, CSV, and Markdown tables. For all currently
selected metrics, lower is better. A positive
`candidate_improvement_percent` means that AIOHMM is better than Gaussian.

The table is a synthetic capability comparison only. It is not evidence of
BMW sensor behaviour or a final real-world model ranking.

## Persisted fitted models

The fitted synthetic Gaussian and AIOHMM files are small and contain model
parameters, feature names, station grid, and configuration—not local file
paths. The repository ignore rules therefore allow only these names below
the curated result tree:

- `gaussian_model.npz`;
- `aiohmm_model.npz`.

Raw generated datasets and arbitrary `.npz` files remain ignored. Keeping the
small fitted models makes later diagnostic re-evaluation possible without a
two-hour AIOHMM refit.
