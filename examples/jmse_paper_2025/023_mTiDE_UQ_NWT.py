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
from nwt_mask_helpers import nwtTiDEMasks, maskFunction
from matplotlib import pyplot as plt
from darts.utils.likelihood_models.torch import LaplaceLikelihood
import seaborn as sns
from darts import concatenate
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

# Initialize masking
cov_array = cov_series.all_values(copy=True)  # shape: [T, C_cov, 1]
tgt_array = tgt_series.all_values(copy=True)  # shape: [T, C_tgt, 1]

# Merge cov + tgt temporarily to work with the full array
data_array = np.concatenate([cov_array, tgt_array], axis=1)

# Define index mappings
cov_idx = list(range(cov_array.shape[1]))
tgt_idx = list(range(cov_array.shape[1], cov_array.shape[1] + tgt_array.shape[1]))

# Define groups, e.g. A, B, C, D (indices in original data_array)
group_indices = {
    "A": [0, 1, 2, 3, 4, 5],     # example indices
    "B": [6, 7, 8, 9, 10, 11],
    "C": [12, 13, 14, 15, 16, 17],
    "D": [18, 19, 20, 21, 22, 23],
}
group_to_idx = {k: i for i, k in enumerate(group_indices.keys())}

masked_cov_array, masked_tgt_array = maskFunction(
    data_array=data_array,
    cov_idx=cov_idx,
    tgt_idx=tgt_idx,
    group_indices=group_indices,
    group_to_idx=group_to_idx
)

# Build TimeSeries
cov_masked = TimeSeries.from_times_and_values(
    cov_series.time_index,
    masked_cov_array.squeeze(-1),  # shape [T, C_cov + 4] if 4 switches
    columns=list(cov_series.components) + ["switch_A", "switch_B", "switch_C", "switch_D"]
)

tgt_masked = TimeSeries.from_times_and_values(
    tgt_series.time_index,
    masked_tgt_array.squeeze(-1),
    columns=tgt_series.components
)

# Split the data into training and validation sets
tgt_train, tgt_val = tgt_masked.split_after(0.8)
cov_train, cov_val = cov_masked.split_after(0.8)

# Scale the data
scaler_tgt = Scaler()
scaler_cov = Scaler()

tgt_train_scaled = scaler_tgt.fit_transform(tgt_train)
tgt_val_scaled = scaler_tgt.transform(tgt_val)
cov_train_scaled = scaler_cov.fit_transform(cov_train)
cov_val_scaled = scaler_cov.transform(cov_val)


# for component in ['p1', 'p7', 'p15', 'switch_A', 'switch_B', 'switch_C', 'switch_D']:
#     cov_train_masked[component].plot(label=f"Masked Covariate: {component}")

# plt.legend()
# plt.show()

# Create the RNN (LSTM-based) model
from darts.models import TiDEModel

modelName = f"baseline_model_NWT_mtide_n{n}_m{m}_e{e}"

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
# _, _, minTaus, allTaus, mostFrequentminTau = rolling_probabilistic_forecast(
#     model,
#     series=mask.tgt,
#     past_covariates=mask.cov,
#     input_chunk_length=m,
#     forecast_horizon=n,
#     stride=int(n/10),
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

# plt.show()

