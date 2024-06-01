import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from vmod.dolphinn import DOLPHINN as DOL
from sklearn.metrics import mean_absolute_error

"""
In this example:
- load a trained WRP model using DOLPHINN on new testseed wave data (IR4)
"""

# Configure
TEST = "2aa"
DOLPHINN_PATH = os.path.join("..", "saved_models", "1d", "IR4_model")
DATA_PATH = os.path.join("..", "data", "FOCAL_wavedata", "scaledup", "IR-4_testseed.csv")
START_TIME = 100
PRESENT_TIME = 9000

if not os.path.exists(os.path.join("..", "figures", f"{TEST}")):
    os.makedirs(os.path.join("..", "figures", f"{TEST}"))

# call dolphinn
dol = DOL()
dol.load(DOLPHINN_PATH)

# predict
data = pd.read_csv(DATA_PATH)
t0 = START_TIME
t1 = PRESENT_TIME
t2 = dol.time_horizon
t0_idx = np.where(np.min(np.abs(data['Time'] - t0)) == np.abs(data['Time'] - t0))[0][0]
t1_idx = np.where(np.min(np.abs(data['Time'] - t1)) == np.abs(data['Time'] - t1))[0][0]
t2_idx = np.where(np.min(np.abs(data['Time']-(t2+t1))) == np.abs(data['Time']-(t2+t1)))[0][0]
past_wave = data[dol.dof].mul(dol.conversion, axis=1).iloc[0:t1_idx]
time = data['Time'].iloc[0:t2_idx]
t_pred, y_hat = dol.wrp_predict(time, past_wave, history=1e4)
t0_hat_idx = np.where(np.min(np.abs(t_pred - t0)) == np.abs(t_pred - t0))[0][0]
t1_hat_idx = np.where(np.min(np.abs(t_pred - t1)) == np.abs(t_pred - t1))[0][0]
plt.figure(figsize=(10, 5))
plt.plot(time.iloc[0:t1_idx], past_wave["wave5"][0:t1_idx], color='black', label='Actual')
plt.scatter(t_pred, y_hat["wave5"], color='red', linestyle='-', label='Predicted')
plt.xlim((t1-250, t1+50))
plt.legend()
# plt.savefig(fr"..\figures\{TEST}\test.pdf", format="pdf")

# Compute error
mae = mean_absolute_error(past_wave["wave5"][t0_idx:t1_idx], y_hat["wave5"][t0_hat_idx:t1_hat_idx])