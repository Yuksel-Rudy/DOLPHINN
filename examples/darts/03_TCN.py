import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

fielPath = os.path.dirname(os.path.abspath(__file__))   
dataPath = os.path.join(fielPath, "waveData.csv")

# Read the CSV file
waveData = pd.read_csv(dataPath)
time = waveData["Time"].to_numpy()
prob = waveData.iloc[:, 1:].to_numpy()

# Interpolate prob data to 2.0Hz time
time = np.arange(time[0], time[-1], 0.5)
prob_interp = np.zeros((len(time), prob.shape[1]))
for i in range(prob.shape[1]):
    prob_interp[:, i] = np.interp(time, waveData["Time"], prob[:, i])

start_time = pd.Timestamp("2025-01-01")
datetime_index = start_time + pd.to_timedelta(time, unit="s")
# Create a pandas DataFrame
df_interp = pd.DataFrame(prob_interp, index=datetime_index)
df_interp.index.name = "Time"

# Create TimeSeries
ts = TimeSeries.from_dataframe(df_interp)

train, val = ts.split_after(0.75)

# Plot the training and validation sets
train.plot(label="Train")
val.plot(label="Validation")
plt.title("Training and Validation Sets")
plt.legend()
plt.show()

scaler = Scaler()

# Scale Data
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)
ts_scaled = scaler.transform(ts)

# Fit TCN model
from darts.models import TCNModel
model = TCNModel(input_chunk_length=30,
                 output_chunk_length=10,
                 n_epochs=500,
                 dropout=0.1,
                 kernel_size=5,
                 num_filters=3, 
                 weight_norm=True, 
                 save_checkpoints=True, 
                 random_state=42)

model.fit(series=train_scaled)

