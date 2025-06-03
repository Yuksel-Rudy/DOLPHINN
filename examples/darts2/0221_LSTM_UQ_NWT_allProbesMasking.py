from darts.models import BlockRNNModel
from darts.utils.likelihood_models import LaplaceLikelihood

import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from helpers import (
    interpolateData, split_ts, mae_hist,
    demean_series, uncertainty, get_predictionHorizon, 
    rolling_probabilistic_forecast, print_tau_summary
)
from plot_helpers import (plot_val, plot_prediction_histogram, 
                          plot_violin_tau_distribution, plot_tau_heatmap)
from nwt_mask_helpers import nwtLSTMMasks
from matplotlib import pyplot as plt
from darts.utils.likelihood_models.torch import LaplaceLikelihood
import seaborn as sns
from darts.metrics import mae
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

n = int(predictionHorizon/dt)
m = n*2
e = 1000

# Mask settings
prbN = 10  # number of probes to compute MAE for in the end
rhos = [0.0, 0.25, 0.5, 0.75]  # randomness thresholds
dlta = 25 # delta in seconds for the mask
dlta = int(dlta / dt)  # Convert delta from seconds to number of timesteps
nDivs = 4

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
maeValues = np.zeros((prbN, len(rhos)))
for i, rho in enumerate(rhos):
    mask = nwtLSTMMasks(nComponents=nComponents, nDivs=nDivs, tgt=val, maskFactor=0.0)
    mask.maskAll(threshold=rho, window_size=dlta, mask_type="substitute")
    mask.tgt = scaler_tgt.transform(mask.tgt)  # Scale the target series after masking

    # Create the RNN (LSTM-based) model
    modelName = f"baseline_model_NWT_lstm_n{n}_m{m}_e{e}"

    # Check if the model already exists
    if os.path.exists(os.path.join(modelPath, modelName)):
        model = BlockRNNModel.load(os.path.join(modelPath, modelName))
    else:
        raise ValueError(f"Model {modelName} does not exist. Please train the model first.")


    # Backtesting [baseline]
    # tau = np.mean(minTaus)
    # if int(tau / dt) == 0:
    #     tau = dt
    backtest_scaled = model.historical_forecasts(
        mask.tgt,
        forecast_horizon=int(33 / dt),  # n, int(33 / dt)
        verbose=True,
        retrain=False
    )
    backtest = scaler_tgt.inverse_transform(backtest_scaled)
    # Calculate the Mean Absolute Error (MAE)

    mae_values = {}
    for comp in val.components[-prbN:]:
        mae_values[comp] = mae(val[comp], backtest[comp])
    maeValues[:, i] = np.array(list(mae_values.values()))


# save the mae Values to csv file
mae_df = pd.DataFrame(maeValues, columns=rhos, index=val.components[-prbN:])
mae_df.to_csv(os.path.join(filePath, f"LSTM_UQ_NWT_allProbesMasking_mae_values_masking_{dlta}s.csv"))

# Plot the maeValues as a heatmap [temporal mask and randomness threshold heetmap]
# plt.figure(figsize=(12, 8))
# sns.heatmap(maeValues.mean(axis=0), annot=True, fmt=".2f", cmap="viridis",
#             xticklabels=dlts, yticklabels=rhos)
# plt.xlabel(r"masking window, $\Delta_m$ (s)")
# plt.ylabel(r"randomness threshold, $\rho$")
# plt.show()