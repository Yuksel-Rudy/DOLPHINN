from darts.models import BlockRNNModel
from darts.utils.likelihood_models import LaplaceLikelihood

import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from vmod.masking.mask_helpers import (
    interpolateData, split_ts, mae_hist,
    demean_series, uncertainty, get_predictionHorizon, 
    rolling_probabilistic_forecast, print_tau_summary
)
from vmod.masking.mask_plot_helpers import (plot_val, plot_prediction_histogram, 
                          plot_violin_tau_distribution, plot_tau_heatmap)
from vmod.masking.mask import nwtLSTMMasks
from matplotlib import pyplot as plt
from darts.utils.likelihood_models.torch import LaplaceLikelihood
import seaborn as sns
from darts.metrics import mae

# Paths
filePath = os.path.join(os.path.dirname(__file__))
dataPath = os.path.join(filePath, "data")   
modelPath = os.path.join(filePath, "models", "baseline_model_NWT_LSTM_UQ")


# Load the data
dataset = "IR-1_NWT"
df = pd.read_csv(os.path.join(dataPath, f"{dataset}.csv"))

# Model settings
dt = 0.5  # timestep in seconds
predictionHorizon = 60
prbN = 15  # number of probes to conduct trimming analysis on

n = int(predictionHorizon/dt)
m = n*2
e = 1000

# Mask settings
maskMode = "allProbes"  # Options: "None", "allProbes", "substitute", "A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD", "ABCD"
nDivs = 4
rho = 0.25
dlta = 25  # delta in seconds for the mask
dlta = int(dlta / dt)  # Convert delta from seconds to number of timesteps

# Interpolate the data
ts = interpolateData(df, dt=dt)
nComponents = ts.shape[1]-1  # Number of components in the time series (excluding time)

# Split the data into training and validation sets
train, val = ts.split_after(0.8)

# Scale the data
scaler_tgt = Scaler()
scaler_cov = Scaler()

train_scaled = scaler_tgt.fit_transform(train)
val_scaled = scaler_tgt.transform(val)

# initialize masks and apply masks
mask = nwtLSTMMasks(nComponents=nComponents, nDivs=nDivs, tgt=val, maskFactor=0.0)

if maskMode == "allProbes" or maskMode == "substitute_LSTM":
    mask.maskAll(threshold=rho, window_size=dlta)   
elif maskMode == "ramp":
    mask.maskAll(threshold=rho, window_size=dlta, mask_type="ramp")
elif maskMode != "None":
    for letter in maskMode:
        getattr(mask, f"mask{letter}")()

mask.tgt = scaler_tgt.transform(mask.tgt)  # Scale the target series after masking

# Create the RNN (LSTM-based) model
modelName = f"baseline_model_NWT_lstm_n{n}_m{m}_e{e}"

# Check if the model already exists
if os.path.exists(os.path.join(modelPath, modelName)):
    model = BlockRNNModel.load(os.path.join(modelPath, modelName))
else:
    model = BlockRNNModel(
        model="LSTM",
        input_chunk_length=m,
        output_chunk_length=n,
        n_epochs=e,
        n_rnn_layers=3,
        batch_size=32,
        hidden_dim=100,
        dropout=0.1,
        likelihood=LaplaceLikelihood(),
        optimizer_kwargs={"lr": 1e-3},
        random_state=None,
        model_name=modelName,
        save_checkpoints=True,
    )
    model.fit(train_scaled, verbose=True)

    # Save the model
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    model.save(os.path.join(modelPath, modelName))

# Forecasting
forecast_scaler = model.predict(
n=n,
series=mask.tgt,  # conditioning on this history
num_samples=1000  # optional if using quantiles; required if using likelihoods like GaussianLikelihood
)
forecast = scaler_tgt.inverse_transform(forecast_scaler)
original = scaler_tgt.inverse_transform(mask.tgt)
originalO = scaler_tgt.inverse_transform(val_scaled)
plot_val(
    original,
    forecast,
    original.components,
    mask_log=mask.maskLog
)
# Backtesting with sampling
# _, _, minTaus, allTaus, mostFrequentminTau = rolling_probabilistic_forecast(
#     model,
#     series=val_scaled,
#     past_covariates=None,  # No covariates for LSTM model
#     input_chunk_length=m,
#     forecast_horizon=n,
#     components=mask.tgt.components[-prbN:],
#     stride=int(n),
#     num_samples=1000,
#     delta_0='solve_for'
# )
# # Plot the histogram of prediction horizons
# plot_prediction_histogram(minTaus)
# # Plot the violin plot of tau distribution
# plot_violin_tau_distribution(allTaus, n, dt)
# # Print the results
# print_tau_summary(minTaus, mostFrequentminTau)

# Backtesting
# tau = np.mean(minTaus)
# if int(tau / dt) == 0:
#     tau = dt
backtest_scaled = model.historical_forecasts(
    mask.tgt,
    forecast_horizon=n,
    verbose=True,
    retrain=False
)
backtest = scaler_tgt.inverse_transform(backtest_scaled)
plot_val(
    original,
    backtest,
    original.components,
    mask_log=mask.maskLog
)



# Calculate the Mean Absolute Error (MAE)
from darts.metrics import mae
mae_values = {}
for comp in val.components[-prbN:]:
    mae_values[comp] = mae(val[comp], backtest[comp])

# Print results
for comp, err in mae_values.items():
    print(f"{comp}: MAE = {err:.4f}")


if maskMode == "substitute_LSTM":
    backtest_scaledLv1 = model.historical_forecasts(
        mask.tgt,
        forecast_horizon=10,
        verbose=True,
        retrain=False
    )
    mask.maskAll_LSTM(historical_forecast=backtest_scaledLv1, prbN=5)

    backtestLv2_scaled = model.historical_forecasts(
        mask.tgt,
        forecast_horizon=n,
        verbose=True,
        retrain=False
    )
    backtestLv2 = scaler_tgt.inverse_transform(backtestLv2_scaled)
    originalLv2 = scaler_tgt.inverse_transform(mask.tgt)
    plot_val(
        originalLv2,
        backtestLv2,
        original.components,
        mask_log=mask.maskLog
    )


    mae_values = {}
    for comp in val.components[-prbN:]:
        mae_values[comp] = mae(val[comp], backtestLv2[comp])

    # Print results
    for comp, err in mae_values.items():
        print(f"{comp}: LV2 - MAE = {err:.4f}")

plt.show()