import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

# --- Load and interpolate ---
fielPath = os.path.dirname(os.path.abspath(__file__))   
dataPath = os.path.join(fielPath, "waveData.csv")

waveData = pd.read_csv(dataPath)
time = waveData["Time"].to_numpy()
prob = waveData.iloc[:, 1:].to_numpy()

time = np.arange(time[0], time[-1], 0.5)
prob_interp = np.zeros((len(time), prob.shape[1]), dtype=np.float32)
for i in range(prob.shape[1]):
    prob_interp[:, i] = np.interp(time, waveData["Time"], prob[:, i])

# --- Convert to TimeSeries ---
start_time = pd.Timestamp("2025-01-01")
datetime_index = start_time + pd.to_timedelta(time, unit="s")

# Upstream as covariates: columns 0–20
df_covariates = pd.DataFrame(prob_interp[:, :21], index=datetime_index)
df_covariates.index.name = "Time"
past_covariates = TimeSeries.from_dataframe(df_covariates)

# Downstream as target: columns 21+
df_target = pd.DataFrame(prob_interp[:, 21:], index=datetime_index)
df_target.index.name = "Time"
target_series = TimeSeries.from_dataframe(df_target)

# --- Train/Validation split ---
train_target, val_target = target_series.split_after(0.75)
train_cov, val_cov = past_covariates.split_after(0.75)

# --- Scale data ---
scaler_target = Scaler()
scaler_cov = Scaler()

train_target_scaled = scaler_target.fit_transform(train_target)
val_target_scaled = scaler_target.transform(val_target)
full_target_scaled = scaler_target.transform(target_series)

train_cov_scaled = scaler_cov.fit_transform(train_cov)
val_cov_scaled = scaler_cov.transform(val_cov)
full_cov_scaled = scaler_cov.transform(past_covariates)

from darts.models import TCNModel

model = TCNModel(
    input_chunk_length=30,
    output_chunk_length=10,
    n_epochs=500,
    dropout=0.1,
    kernel_size=5,
    num_filters=3,
    weight_norm=True,
    save_checkpoints=True,
    random_state=42
)

model.fit(series=train_target_scaled, past_covariates=train_cov_scaled, verbose=True)

forecast = model.predict(n=len(val_target_scaled), past_covariates=val_cov_scaled)

# Optional: inverse transform to original scale
forecast_orig = scaler_target.inverse_transform(forecast)
val_orig = scaler_target.inverse_transform(val_target_scaled)

# Plot
val_orig.plot(label="actual")
forecast_orig.plot(label="forecast")
plt.legend()
plt.title("Forecast vs Actual (Downstream Probes)")
plt.show()
