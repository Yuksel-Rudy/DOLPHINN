from darts.models import TiDEModel
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
from vmod.masking.mask import nwtLSTMMasks, nwtTiDEMasks
from matplotlib import pyplot as plt
from darts.utils.likelihood_models.torch import LaplaceLikelihood
import seaborn as sns
from darts.metrics import mae
from darts.metrics import mae
from scipy.stats import t

# Paths
filePath = os.path.join(os.path.dirname(__file__))
dataPath = os.path.join(filePath, "data")   
modelPath = os.path.join(filePath, "models", "baseline_model_tide_UQ")

# Load the data
dataset = "IR-4"
df = pd.read_csv(os.path.join(dataPath, f"{dataset}.csv"))

# Model settings
dt = 0.5  # timestep in seconds
predictionHorizon = 60  # in seconds

n = int(predictionHorizon/dt)
m = n*2
e = 1000

# Mask settings
taus = [34, 20]  # [34, 20] [41, 24]
rho  = 0.25
dlts = [25, 250]
n_repeat = 25
nDivs = 3

# Interpolate the data
ts = interpolateData(df, dt=dt)
nComponents = ts.shape[1]-1  # Number of components in the time series (excluding time)

cov_idx = [0, 1, 2, 3]
tgt_idx = [4]

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

mae_store = np.zeros((n_repeat, len(taus), len(dlts)))

# Pull up the model
modelName = f"baseline_model_tide_n{n}_m{m}_e{e}_w{dataset}"
# Check if the model already exists
if os.path.exists(os.path.join(modelPath, modelName)):
    model = TiDEModel.load(os.path.join(modelPath, modelName))
else:
    raise ValueError(f"Model {modelName} does not exist. Please train the model first.")
for r in range(n_repeat):
    for i, tau in enumerate(taus):
        for j, dlta in enumerate(dlts):
            mask = nwtTiDEMasks(nComponents=nComponents, nDivs=nDivs, cov=cov_val, tgt=tgt_val, maskFactor=0.0)
            mask.maskAll(threshold=rho, window_size=dlta, mask_type="factor")
            mask.tgt = scaler_tgt.transform(mask.tgt)
            mask.cov = scaler_cov.transform(mask.cov)

            backtest_scaled = model.historical_forecasts(
            mask.tgt,
            past_covariates=mask.cov,
            forecast_horizon=int(tau / dt),  # n, int(44 / dt)
            verbose=True,
            retrain=False
            )
            backtest = scaler_tgt.inverse_transform(backtest_scaled)
            mae_values = mae(tgt_val, backtest)
            mae_store[r, i, j] = mae_values
            print(f"Repeat {r+1}, Tau {tau}, Delta {dlta}: MAE = {mae_values}")

# Calculate the mean and 95% CI 
mae_mean = np.mean(mae_store, axis=0)
mae_std = np.std(mae_store, axis=0, ddof=1)
ci95 = t.ppf(0.975, df=n_repeat-1) * mae_std / np.sqrt(n_repeat)

mae_df = pd.DataFrame(mae_mean, columns=dlts, index=taus)
ci_df  = pd.DataFrame(ci95, columns=dlts, index=taus)



plt.rcParams.update({'font.size': 16})
taus_str = [f"{t} s" for t in taus]
n_groups = len(taus)
n_bars = len(dlts)

bar_width = 0.35
index = np.arange(n_groups)

colors = plt.cm.magma(np.linspace(0.2, 0.8, n_bars))

fig, ax = plt.subplots(figsize=(5, 5))

for j, dlt in enumerate(dlts):
    means = mae_df[dlt].values
    errors = ci_df[dlt].values
    bar_positions = index + j * bar_width
    ax.bar(bar_positions, means, bar_width, yerr=errors, label=fr"$\Delta t_{{msk}}$={dlt} s",
           capsize=5, color=colors[j], alpha=0.8)

# Formatting
ax.set_ylabel("MAE (m)")
ax.set_xlabel("model type")
ax.set_xticks(index + bar_width * (n_bars - 1) / 2)
ax.set_xticklabels([fr"moderate: $\tau = {taus[0]} s$", fr"conservative: $\tau = {taus[1]} s$"])
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle=':', linewidth=0.5)
ax.legend(loc="upper right")
fig.tight_layout()
plt.show()