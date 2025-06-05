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
from vmod.masking.mask import nwtTiDEMasks
from matplotlib import pyplot as plt
from darts.utils.likelihood_models.torch import LaplaceLikelihood
import seaborn as sns
from darts.metrics import mae
from darts.metrics import mae
from scipy.stats import t

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
prbN = 6  # number of probes to compute MAE for in the end
rhos = [0.0, 0.25, 0.50, 0.75]  # randomness thresholds
dlta = 25 # delta in seconds for the mask
dlta = int(dlta / dt)  # Convert delta from seconds to number of timesteps
nDivs = 4

# rhos = [0.25]
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

# initialize masks and apply masks
# n_repeat = 10
n_repeat = 1
mae_store = np.zeros((n_repeat, prbN, len(rhos)))

for r in range(n_repeat):
    for i, rho in enumerate(rhos):
        mask = nwtTiDEMasks(nComponents=nComponents, nDivs=nDivs, cov=cov_val, tgt=tgt_val, maskFactor=0.0)
        mask.maskAll(threshold=rho, window_size=dlta, mask_type="factor")
        mask.tgt = scaler_tgt.transform(mask.tgt)  # Scale the target series after masking
        mask.cov = scaler_cov.transform(mask.cov)  # Scale the covariates after masking

        # mask.cov.plot()
        # plt.show()

        # Create the TiDE model
        modelName = f"baseline_model_NWT_tide_n{n}_m{m}_e{e}"

        # Check if the model already exists
        if os.path.exists(os.path.join(modelPath, modelName)):
            model = TiDEModel.load(os.path.join(modelPath, modelName))
        else:
            raise ValueError(f"Model {modelName} does not exist. Please train the model first.")


        backtest_scaled = model.historical_forecasts(
            mask.tgt,
            past_covariates=mask.cov,
            forecast_horizon=n,  # n, int(44 / dt)
            verbose=True,
            retrain=False
        )
        backtest = scaler_tgt.inverse_transform(backtest_scaled)
        # Calculate the Mean Absolute Error (MAE)

        mae_values = {}
        for comp in tgt_val.components[-prbN:]:
            mae_values[comp] = mae(tgt_val[comp], backtest[comp])

        mae_store[r, :, i] = np.array(list(mae_values.values()))




# Mean and 95% CI for each [probe, rho]
mae_mean = np.mean(mae_store, axis=0)       # shape: [prbN, len(rhos)]
mae_std = np.std(mae_store, axis=0, ddof=1)
ci95 = t.ppf(0.975, df=n_repeat-1) * mae_std / np.sqrt(n_repeat)

# Convert to DataFrame for plotting
probes = tgt_val.components[-prbN:]
mae_df = pd.DataFrame(mae_mean, columns=rhos, index=probes)
ci_df  = pd.DataFrame(ci95,  columns=rhos, index=probes)

mae_df.to_csv(os.path.join(filePath, f"CI_TiDE_MAE_mean_dlta.csv"))
ci_df.to_csv(os.path.join(filePath, f"CI_TiDE_MAE_ci95_dlta.csv"))


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
colors = plt.cm.magma(np.linspace(0.2, 0.8, len(rhos)))

plt.figure(figsize=(7, 4.5))
for i, rho in enumerate(rhos):
    mean = mae_df[rho]
    ci = ci_df[rho]

    plt.plot(probes, mean, label=f"{rho}", color=colors[i], linewidth=2)
    plt.fill_between(probes, mean - ci, mean + ci, color=colors[i], alpha=0.3, linewidth=0)

plt.ylabel("MAE (m)")
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.xlabel("components")
plt.grid(axis='x', linestyle=':', linewidth=0.5)
# plt.legend(title=r"$\rho$", loc="upper right")
plt.tight_layout()
plt.show()
