# Synthetic model comparison

Positive improvement means the candidate is better. These synthetic results are capability checks, not final BMW-data rankings.

Baseline: `conditional_multivariate_gaussian`  
Candidate: `autoregressive_input_output_hmm`

| Scenario | Metric | Baseline | Candidate | Improvement [%] | Better |
|---|---|---:|---:|---:|---|
| conditional_gaussian | Predictive-mean RMSE | 0.0593421 | 0.0593598 | -0.0296901 | conditional_multivariate_gaussian |
| conditional_gaussian | CRPS | 0.0294127 | 0.0294132 | -0.00196961 | conditional_multivariate_gaussian |
| conditional_gaussian | Energy score | 0.0397456 | 0.0396963 | 0.124043 | autoregressive_input_output_hmm |
| conditional_gaussian | Marginal JS distance | 0.0115355 | 0.0147663 | -28.0076 | conditional_multivariate_gaussian |
| conditional_gaussian | First-difference JS distance | 0.0115174 | 0.0143299 | -24.4202 | conditional_multivariate_gaussian |
| conditional_gaussian | Lag-one correlation MAE | 0.00509823 | 0.00431084 | 15.4443 | autoregressive_input_output_hmm |
| conditional_gaussian | Spatial-correlation RMSE | 0.00484067 | 0.0422544 | -772.903 | conditional_multivariate_gaussian |
| conditional_gaussian | Absolute q99 error | 0.000842459 | 0.00360394 | -327.789 | conditional_multivariate_gaussian |
| latent_autoregressive | Predictive-mean RMSE | 0.109712 | 0.109633 | 0.0722391 | autoregressive_input_output_hmm |
| latent_autoregressive | CRPS | 0.0510161 | 0.0491126 | 3.73103 | autoregressive_input_output_hmm |
| latent_autoregressive | Energy score | 0.0710538 | 0.0685037 | 3.58887 | autoregressive_input_output_hmm |
| latent_autoregressive | Marginal JS distance | 0.165275 | 0.0147578 | 91.0708 | autoregressive_input_output_hmm |
| latent_autoregressive | First-difference JS distance | 0.353916 | 0.0134018 | 96.2133 | autoregressive_input_output_hmm |
| latent_autoregressive | Lag-one correlation MAE | 0.786705 | 0.0043287 | 99.4498 | autoregressive_input_output_hmm |
| latent_autoregressive | Spatial-correlation RMSE | 0.0281299 | 0.0470143 | -67.1329 | conditional_multivariate_gaussian |
| latent_autoregressive | Absolute q99 error | 0.0904198 | 0.00458624 | 94.9278 | autoregressive_input_output_hmm |
| nonlinear_heavy_tailed | Predictive-mean RMSE | 0.110036 | 0.110033 | 0.00308996 | autoregressive_input_output_hmm |
| nonlinear_heavy_tailed | CRPS | 0.0538543 | 0.0534756 | 0.703115 | autoregressive_input_output_hmm |
| nonlinear_heavy_tailed | Energy score | 0.0738688 | 0.0733006 | 0.769172 | autoregressive_input_output_hmm |
| nonlinear_heavy_tailed | Marginal JS distance | 0.0568618 | 0.0232947 | 59.0328 | autoregressive_input_output_hmm |
| nonlinear_heavy_tailed | First-difference JS distance | 0.327047 | 0.0141826 | 95.6634 | autoregressive_input_output_hmm |
| nonlinear_heavy_tailed | Lag-one correlation MAE | 0.784884 | 0.00389005 | 99.5044 | autoregressive_input_output_hmm |
| nonlinear_heavy_tailed | Spatial-correlation RMSE | 0.0147051 | 0.07969 | -441.921 | conditional_multivariate_gaussian |
| nonlinear_heavy_tailed | Absolute q99 error | 0.0457805 | 0.0112047 | 75.5252 | autoregressive_input_output_hmm |
