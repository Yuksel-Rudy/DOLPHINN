import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from helpers import interpolateData, split_ts, plot_val, mae_hist
from matplotlib import pyplot as plt

# Paths
filePath = os.path.join(os.path.dirname(__file__))
dataPath = os.path.join(filePath, "data")

# Load the data
dataset = "IR-1_NWT"
df = pd.read_csv(os.path.join(dataPath, f"{dataset}.csv"))

# Model settings
m = 24
n = 12
e = 100
# Interpolate the data
ts = interpolateData(df, dt=0.5)

cov_idx = list(range(21))
tgt_idx = list(range(21, 25))

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

# Create the baseline model
from darts.models import TCNModel


modelPath = os.path.join(filePath, "models", "baseline_model_NWT")
modelName = f"baseline_model_NWT_n{n}_m{m}_e{e}"
# Check if the model already exists
if os.path.exists(os.path.join(modelPath, modelName)):
    model_to_be_saved = False    
    model = TCNModel.load(os.path.join(modelPath, modelName))
else:
    model_to_be_saved = True
    model = TCNModel(
        input_chunk_length=m,
        output_chunk_length=n,
        n_epochs=e,
        batch_size=32,
        num_layers=2,
        kernel_size=3,
        dropout=0.1,
    )
    # Fit the model
    model.fit(
        tgt_train_scaled,
        past_covariates=cov_train_scaled,
        verbose=True,
    )    



# Backtest the model
backtest_scaled = model.historical_forecasts(
    tgt_val_scaled,
    past_covariates=cov_val_scaled,
    forecast_horizon=n,
    verbose=True,
    retrain=False
)

# Inverse transform the predictions
backtest = scaler_tgt.inverse_transform(backtest_scaled)

# Plot the results
component_of_interest = list(tgt_val.components[-4:])
plot_val(tgt_val, backtest, component_of_interest)
from darts.metrics import mae
# Calculate and print the MAE
mae_value = mae(tgt_val[component_of_interest], backtest[component_of_interest])
print(f"Mean Absolute Error (MAE): {mae_value:.2f}")

# Plot the error histogram
mae_hist(tgt_val[component_of_interest].slice_intersect(backtest[component_of_interest]).values(), backtest[component_of_interest].values())

# Save the model
if model_to_be_saved:
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    model.save(os.path.join(modelPath, modelName))

plt.show()