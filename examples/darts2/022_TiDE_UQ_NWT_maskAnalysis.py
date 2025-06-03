import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from helpers import (
    interpolateData, split_ts, mae_hist, mask_ts,
    demean_series, uncertainty, get_predictionHorizon, 
    rolling_probabilistic_forecast, print_tau_summary
)
from plot_helpers import (plot_val, plot_prediction_histogram, 
                          plot_violin_tau_distribution, plot_tau_heatmap)
from nwt_mask_helpers import nwtTiDEMasks
from matplotlib import pyplot as plt
from darts.utils.likelihood_models.torch import LaplaceLikelihood
import seaborn as sns
from darts.metrics import mae

# Paths
filePath = os.path.join(os.path.dirname(__file__))
dataPath = os.path.join(filePath, "data")   
modelPath = os.path.join(filePath, "models", "baseline_model_NWT_tide_UQ")

# Load the data
dataset = "IR-1_NWT"
df = pd.read_csv(os.path.join(dataPath, f"{dataset}.csv"))

# Model settings
dt = 0.5  # timestep in seconds
predictionHorizon = 60

n = int(predictionHorizon/dt)
m = n*2
e = 1000

nDivs = 4

# Interpolate the data
ts = interpolateData(df, dt=dt)
nComponents = ts.shape[1]-1  # Number of components in the time series (excluding time)

cov_idx = range(0, int(nComponents - nComponents/nDivs))
tgt_idx = range(int(nComponents - nComponents/nDivs), nComponents)

# Split the time series into covariates and target
tgt_series, cov_series = split_ts(ts, cov_idx=cov_idx, tgt_idx=tgt_idx)

# Split the data into training and validation sets
tgt_train, tgt_val = tgt_series.split_after(0.8)
cov_train, cov_val = cov_series.split_after(0.8)

# Scale the data
scaler_tgt = Scaler()
scaler_cov = Scaler()

tgt_train_scaled = scaler_tgt.fit_transform(tgt_train)
tgt_val_scaled = scaler_tgt.transform(tgt_val)
cov_train_scaled = scaler_cov.fit_transform(cov_train)
cov_val_scaled = scaler_cov.transform(cov_val)


# Create the RNN (LSTM-based) model
from darts.models import TiDEModel

modelName = f"baseline_model_NWT_tide_n{n}_m{m}_e{e}"

# Check if the model already exists
if os.path.exists(os.path.join(modelPath, modelName)):
    model = TiDEModel.load(os.path.join(modelPath, modelName))
else:
    raise ValueError(f"Model {modelName} does not exist. Please train the model first.")


mask_modes = [
    "None", "A", "B", "C", "D", "AB", "AC", "AD",
    "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD", "ABCD"
]


results = []

for maskMode in mask_modes:
    print(f"\n--- Processing mask mode: {maskMode} ---")

    # Initialize and apply masks
    mask = nwtTiDEMasks(nComponents=nComponents, nDivs=nDivs, cov=cov_val_scaled, tgt=tgt_val_scaled, maskFactor=0.0)

    if maskMode != "None":
        for letter in maskMode:
            getattr(mask, f"mask{letter}")()

    # Forecast with uncertainty estimation
    _, _, minTaus, allTaus, mostFrequentminTau = rolling_probabilistic_forecast(
        model,
        series=mask.tgt,
        past_covariates=mask.cov,
        input_chunk_length=m,
        forecast_horizon=n,
        stride=int(n / 10),
        num_samples=1000,
        delta_0='solve_for'
    )

    # Compute average tau
    tau = np.mean(minTaus)
    if int(tau / dt) == 0:
        tau = dt  # default to minimum sensible value

    # Backtest using average tau
    backtest_scaled = model.historical_forecasts(
        mask.tgt,
        past_covariates=mask.cov,
        forecast_horizon=int(tau / dt),
        verbose=False,
        retrain=False
    )
    backtest = scaler_tgt.inverse_transform(backtest_scaled)
    
    # Demean
    tgt_val_dm = demean_series(tgt_val)
    backtest_dm = demean_series(backtest)

    # Compute MAE
    mae_value = mae(tgt_val_dm, backtest_dm)

    # Store results
    results.append((maskMode, tau, mae_value))

print("\n--- Results Summary ---")
for maskMode, tau, mae_value in results:
    print(f"Mask Mode: {maskMode}, Tau: {tau}, MAE: {mae_value}")

pd.DataFrame(results, columns=["MaskMode", "MeanTau", "MAE"]).to_csv(os.path.join(filePath, "mask_mode_results.csv"), index=False)