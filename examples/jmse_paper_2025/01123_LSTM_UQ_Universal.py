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
train_list = []
val_list = []

for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    # Load the dataset
    df = pd.read_csv(os.path.join(dataPath, f"{dataset}.csv"))
    # Interpolate the data
    ts = interpolateData(df, dt=dt)
    train, val = ts.split_after(0.8)

    train_list.append(train)
    val_list.append(val)

# Concatenate all time series
train = concatenate(train_list, axis=0, ignore_time_axis=True)
val = concatenate(val_list, axis=0, ignore_time_axis=True)

# Mask settings
maskMode = "None"  # Options: "A", "B", "C"

# Scale the data
scaler_tgt = Scaler()

train_scaled = scaler_tgt.fit_transform(train)
val_scaled = scaler_tgt.transform(val)

# Create the RNN (LSTM-based) model
from darts.models import BlockRNNModel

modelPath = os.path.join(filePath, "models", "baseline_model_lstm_UQ_Universal")
modelName = f"baseline_model_lstm_n{n}_m{m}_e{e}"

# Check if the model already exists
if os.path.exists(os.path.join(modelPath, modelName)):
    model_to_be_saved = False
    model = BlockRNNModel.load(os.path.join(modelPath, modelName))
else:
    model_to_be_saved = True
    model = BlockRNNModel(
        model="LSTM", 
        input_chunk_length=m,
        output_chunk_length=n,
        n_epochs=e,
        n_rnn_layers=3,
        batch_size=32,
        hidden_dim=100,
        likelihood=LaplaceLikelihood(),
        dropout=0.1,
        optimizer_kwargs={"lr": 1e-3},
        model_name=modelName,
        save_checkpoints=True,
        random_state=None,
    )
    model.fit(
        train_scaled,
        verbose=True,
    )
    # Save the model
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    model.save(os.path.join(modelPath, modelName))    


forecast_scaler = model.predict(
n=n,
series=val_scaled,
num_samples=1000
)

# Mask settings
maskMode = "None"
nComponents = ts.shape[1]-1
nDivs = 3
rho = 0.25
dlta= 25
# initialize masks and apply masks
mask = nwtLSTMMasks(nComponents=nComponents, nDivs=nDivs, tgt=val, maskFactor=0.0)

if maskMode == "None":
    pass
elif maskMode == "allProbes":
    mask.maskAll(threshold=rho, window_size=dlta)   
    pass

mask.tgt = scaler_tgt.transform(mask.tgt)

# FORECASTING
forecast_scaler = model.predict(
n=n,
series=mask.tgt,  # conditioning on this history
num_samples=1000  # optional if using quantiles; required if using likelihoods like GaussianLikelihood
)
forecast = scaler_tgt.inverse_transform(forecast_scaler)
original = scaler_tgt.inverse_transform(mask.tgt)
plot_val(
    original,
    forecast,
    original.components,
    mask_log=mask.maskLog
)
plt.show()