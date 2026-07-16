# Synthetic Prototype Results

These results are controlled capability experiments using synthetic data.
They are not evidence of BMW sensor behaviour and are not final real-world
model rankings.

Models evaluated:

1. Conditional multivariate Gaussian
2. Autoregressive input-output HMM

Both models use the same train, validation and test data contract and the
same physical-unit evaluation metrics.

The AIOHMM prototype required approximately two hours.

The evaluation used 64 predictive samples. Interval-coverage values are
affected by finite-ensemble quantile estimation. This will be accounted for
before publication-level evaluation.