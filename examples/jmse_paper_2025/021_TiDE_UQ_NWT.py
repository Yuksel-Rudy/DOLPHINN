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
from vmod.masking.mask import nwtLSTMMasks, nwtTiDEMasks
from matplotlib import pyplot as plt
from darts.utils.likelihood_models.torch import LaplaceLikelihood
import seaborn as sns

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

# Mask settings
maskMode = "None"  # Options: "A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", "ABC", "ABD", "ACD", "BCD", "ABCD"


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

# initiaize masks
mask = nwtTiDEMasks(nComponents=nComponents, nDivs=nDivs, cov=cov_val, tgt=tgt_val, maskFactor=0.0)
if maskMode != "None":
    for letter in maskMode:
        getattr(mask, f"mask{letter}")()

mask.cov = scaler_cov.transform(mask.cov)  # Scale the covariates after masking
mask.tgt = scaler_tgt.transform(mask.tgt)  # Scale the target series after masking


# Create the RNN (LSTM-based) model
from darts.models import TiDEModel

modelName = f"baseline_model_NWT_tide_n{n}_m{m}_e{e}"

# Check if the model already exists
if os.path.exists(os.path.join(modelPath, modelName)):
    model = TiDEModel.load(os.path.join(modelPath, modelName))
else:
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

    # Save the model
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    model.save(os.path.join(modelPath, modelName))


# Forecasting
# forecast_scaler = model.predict(
# n=n,
# past_covariates=cov_val_scaled,
# series=tgt_val_scaled,  # conditioning on this history
# num_samples=1000  # optional if using quantiles; required if using likelihoods like GaussianLikelihood
# )
# # Inverse transform the predictions
# forecast = scaler_tgt.inverse_transform(forecast_scaler)
# plot_val(
#     tgt_val,
#     forecast,
#     tgt_val.components
# )



# Backtesting with sampling
_, _, minTaus, allTaus, mostFrequentminTau = rolling_probabilistic_forecast(
    model,
    series=mask.tgt,
    past_covariates=mask.cov,
    input_chunk_length=m,
    forecast_horizon=n,
    stride=int(n/30),  # int(n/10)
    num_samples=1000,
    delta_0='solve_for'
)
# Plot the histogram of prediction horizons
plot_prediction_histogram(minTaus)
# Plot the violin plot of tau distribution
plot_violin_tau_distribution(allTaus, n, dt, components=mask.tgt.components)
# Print the results
print_tau_summary(minTaus, mostFrequentminTau)



# Backtesting
# tau = np.mean(minTaus)
# if int(tau / dt) == 0:
#     tau = dt
# backtest_scaled = model.historical_forecasts(
#     mask.tgt,
#     past_covariates=mask.cov,
#     forecast_horizon=int(tau/dt),
#     verbose=True,
#     retrain=False
# )    
# backtest = scaler_tgt.inverse_transform(backtest_scaled)
# # demean?
# tgt_val = demean_series(tgt_val)
# backtest = demean_series(backtest)
# plot_val(tgt_val,
#          backtest,
#          tgt_val.components)

# # Calculate the Mean Absolute Error (MAE)
# from darts.metrics import mae
# mae_value = mae(tgt_val, backtest)
# print(f"Mean Absolute Error (MAE): {mae_value:.2f}")

plt.show()

