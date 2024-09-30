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

"""
In this example:
- load a trained WRP model using DOLPHINN on new testseed wave data (IR4) (loop)
"""

# Configure
TEST = "wrp_predict_loop"
DOLPHINN_PATH = os.path.join("saved_models", "1d", "IR4_model")
DATA_PATH = os.path.join("data", "FOCAL_wavedata", "scaledup", "IR-4_testseed.csv")
FINAL_TIME = 100

if not os.path.exists(os.path.join("figures", f"{TEST}")):
    os.makedirs(os.path.join("figures", f"{TEST}"))

# call dolphinn
dol = DOL()
dol.load(DOLPHINN_PATH)
data = pd.read_csv(DATA_PATH)
dofs = data[dol.dof].mul(dol.conversion, axis=1)


# time indices: assuming time always starts at zero:
ph = dol.time_horizon  # future time
t2_idx = np.where(np.min(np.abs(data['Time'] - ph)) == np.abs(data['Time'] - ph))[0][0]  # future time index
tf_idx = np.where(np.min(np.abs(data['Time'] - FINAL_TIME)) == np.abs(data['Time'] - FINAL_TIME))[0][0]  # final time index
# initialize time, state, and wave
time = data['Time'].iloc[0:t2_idx].reset_index(drop=True)
past_wave = pd.DataFrame(columns=dofs.columns)
t_preds = pd.DataFrame()
y_preds = pd.DataFrame()


# Run predictions over time
for t_idx in range(tf_idx):
    print(f"present time: {np.round(data['Time'][t_idx],2)}s")
    present_wave = dofs.iloc[t_idx]
    # Append the current state to the state DataFrame
    past_wave = pd.concat([past_wave, pd.DataFrame([present_wave], columns=dofs.columns)], ignore_index=True)

    # Append current values to time and wave if moving through the dataframe
    if t_idx + t2_idx < len(data):
        new_time = data['Time'].iloc[t_idx + t2_idx]
        time = pd.concat([time, pd.Series([new_time])], ignore_index=True)

    # predict
    try:
        t_pred, y_pred = dol.wrp_predict(time, past_wave, history=0)
        # Append to final results:
        t_preds = pd.concat([t_preds, t_pred.iloc[-1:]], ignore_index=True)
        y_preds = pd.concat([y_preds, y_pred.iloc[-1:]], ignore_index=True)
    except Exception as e:
        print(f"Could not produce prediction because of the following error: {e}")
