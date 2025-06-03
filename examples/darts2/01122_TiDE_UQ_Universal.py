import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from matplotlib import pyplot as plt
from darts.utils.likelihood_models.torch import LaplaceLikelihood
from darts import concatenate
from vmod.masking.mask import nwtLSTMMasks, nwtTiDEMasks
from vmod.masking.mask_helpers import (
    interpolateData, split_ts, mae_hist,
    demean_series, uncertainty, get_predictionHorizon, 
    rolling_probabilistic_forecast, print_tau_summary
)
from vmod.masking.mask_plot_helpers import (plot_val, plot_prediction_histogram, 
                          plot_violin_tau_distribution, plot_tau_heatmap)

# Paths
filePath = os.path.join(os.path.dirname(__file__))
dataPath = os.path.join(filePath, "data")

# Model settings
dt = 0.5  # timestep in seconds
predictionHorizon = 60  # in seconds

n = int(predictionHorizon/dt)
m = n*2
e = 1000
cov_idx = [0, 1, 2, 3]
tgt_idx = [4]

# Load the data
datasets = ["IR-1", "IR-2", "IR-3", "IR-4"]
tgt_train_list = []
tgt_val_list = []
cov_train_list = []
cov_val_list = []
for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    # Load the dataset
    df = pd.read_csv(os.path.join(dataPath, f"{dataset}.csv"))
    # Interpolate the data
    ts = interpolateData(df, dt=dt)
    tgt_series, cov_series = split_ts(ts, cov_idx=cov_idx, tgt_idx=tgt_idx)
    tgt_train, tgt_val = tgt_series.split_after(0.8)
    cov_train, cov_val = cov_series.split_after(0.8)

    tgt_train_list.append(tgt_train)
    tgt_val_list.append(tgt_val)
    cov_train_list.append(cov_train)
    cov_val_list.append(cov_val)

# Concatenate all time series
tgt_train = concatenate(tgt_train_list, axis=0, ignore_time_axis=True)
tgt_val = concatenate(tgt_val_list, axis=0, ignore_time_axis=True)
cov_train = concatenate(cov_train_list, axis=0, ignore_time_axis=True)
cov_val = concatenate(cov_val_list, axis=0, ignore_time_axis=True)

# Mask settings
maskMode = "None"  # Options: "A", "B", "C"

# Scale the data
scaler_tgt = Scaler()
scaler_cov = Scaler()

tgt_train_scaled = scaler_tgt.fit_transform(tgt_train)
tgt_val_scaled = scaler_tgt.transform(tgt_val)
cov_train_scaled = scaler_cov.fit_transform(cov_train)
cov_val_scaled = scaler_cov.transform(cov_val)

# Create the RNN (LSTM-based) model
from darts.models import TiDEModel

modelPath = os.path.join(filePath, "models", "baseline_model_tide_UQ_Universal")
modelName = f"baseline_model_tide_n{n}_m{m}_e{e}"

# Check if the model already exists
if os.path.exists(os.path.join(modelPath, modelName)):
    model_to_be_saved = False
    model = TiDEModel.load(os.path.join(modelPath, modelName))
else:
    model_to_be_saved = True
    model = TiDEModel(
        input_chunk_length=m,
        output_chunk_length=n,
        n_epochs=e,
        batch_size=32,
        hidden_size=100,
        likelihood=LaplaceLikelihood(),
        dropout=0.1,
        optimizer_kwargs={"lr": 1e-3},
        random_state=None,
    )
    model.fit(
        tgt_train_scaled,
        past_covariates=cov_train_scaled,
        verbose=True,
    )

# Mask settings
maskMode = "None"
nComponents = ts.shape[1]-1
nDivs = 3
rho = 0.25
dlta= 25
# initialize masks and apply masks
mask = nwtTiDEMasks(nComponents=nComponents, nDivs=nDivs, cov=cov_val, tgt=tgt_val, maskFactor=0.0)

if maskMode == "None":
    pass
elif maskMode == "allProbes":
    mask.maskAll(threshold=rho, window_size=dlta)   
    pass

mask.cov = scaler_cov.transform(mask.cov)
mask.tgt = scaler_tgt.transform(mask.tgt)

# FORECASTING
# forecast_scaler = model.predict(
# n=n,
# past_covariates=mask.cov,  # conditioning on this history
# series=mask.tgt,  # conditioning on this history
# num_samples=1000  # optional if using quantiles; required if using likelihoods like GaussianLikelihood
# )
# forecast = scaler_tgt.inverse_transform(forecast_scaler)
# original = scaler_tgt.inverse_transform(mask.tgt)
# plot_val(
#     original,
#     forecast,
#     original.components,
#     mask_log=mask.maskLog
# )
# plt.show()

# BACKTESTING WITH SAMPLING
# _, _, minTaus, allTaus, mostFrequentminTau = rolling_probabilistic_forecast(
#     model,
#     series=mask.tgt,
#     past_covariates=mask.cov,
#     input_chunk_length=m,
#     forecast_horizon=n,
#     stride=int(n/10),  # int(n/10)
#     num_samples=1000,
#     delta_0='solve_for'
# )
# # Plot the histogram of prediction horizons
# plot_prediction_histogram(minTaus)
# # Plot the violin plot of tau distribution
# plot_violin_tau_distribution(allTaus, n, dt)
# # Print the results
# print_tau_summary(minTaus, mostFrequentminTau)

# plt.show()

# BACKTESTING
backtest_scaled = model.historical_forecasts(
    mask.tgt,
    past_covariates=mask.cov,
    forecast_horizon=int(39.00/dt),
    verbose=True,
    retrain=False
)    
backtest = scaler_tgt.inverse_transform(backtest_scaled)
plot_val(tgt_val,
         backtest,
         tgt_val.components)

# Calculate the Mean Absolute Error (MAE)
from darts.metrics import mae
mae_value = mae(tgt_val, backtest)
print(f"Mean Absolute Error (MAE): {mae_value:.2f}")

plt.show()
