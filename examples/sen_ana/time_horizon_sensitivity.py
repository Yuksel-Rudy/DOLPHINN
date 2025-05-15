import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import os
import sys
from pathlib import Path
from vmod.dolphinn import DOLPHINN as DOL
from vmod.p2v import get_psd
import numpy as np

"""
In this example: We will vary the time horizon of the MLSTM model to see how it affects the prediction
"""

plt.rcParams.update({'font.size': 18})
matplotlib.rcParams['font.family'] = 'Times New Roman'

test = "sen_ana"
wave = "IR1"
time_horizon = [80]  # in seconds

config_file_path = os.path.join("dol_input", "sen_ana", f"{wave}_tau_x.yaml")
if not os.path.exists(os.path.join("figures", f"{test}")):
    os.makedirs(os.path.join("figures", f"{test}"))

# call dolphinn

maes = []
for th in time_horizon:
    dol = DOL(config_path=config_file_path)
    dol.time_horizon = th
    dol.m = int(np.round(dol.time_horizon / dol.timestep, 0))  # corresponding to TIME_HORIZON
    dol.n = int(np.round(dol.nm * dol.m))
    dol.future_lower_lim = dol.m
    dol.train()
    r_square, mae, y, y_hat = dol.test()
    t = np.linspace(0, (y.shape[0]-1)*dol.timestep, y.shape[0])
    maes.append(mae)


    # post-processing (TD)
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(len(dol.labels), 1)
    for i, (label, unit) in enumerate(zip(dol.labels, dol.unit)):
        ax = plt.subplot(gs[i])
        ax.plot(t, y[:, i], label='experiment', color='black')
        ax.plot(t, y_hat[:, i], label='DOLPHINN', color='red', linestyle='-')
        ax.set_xlabel('t [s]')
        ax.set_ylabel(f"wave elevation {unit}")
        ax.set_xlim((1000, 1150))
        ax.legend(loc='upper right')
        ax.grid()
        ax.set_title(f"{wave} - Time Horizon: {th}")
        
        # Calculate the range of y and y_hat
        y_range = np.max(y[:, i]) - np.min(y[:, i])
        y_hat_range = np.max(y_hat[:, i]) - np.min(y_hat[:, i])
        relative_size = y_hat_range / y_range
        ax.text(0.5, 0.9, f"Relative Size: {relative_size:.2f}", transform=ax.transAxes, ha='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"{test}", f"{wave}_TD_{th}.pdf"), format="pdf")
    for i, (label, unit) in enumerate(zip(dol.labels, dol.unit)):
        ax = plt.subplot(gs[i])
        ax.plot(t, y[:, i], label='experiment', color='black')
        ax.plot(t, y_hat[:, i], label='DOLPHINN', color='red', linestyle='-')
        ax.set_xlabel('t [s]')
        ax.set_ylabel(f"wave elevation {unit}")
        ax.set_xlim((1000, 1150))
        ax.legend(loc='upper right')
        ax.grid()
        ax.set_title(f"{wave} - Time Horizon: {th}")

    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"{test}", f"{wave}_TD_{th}.pdf"), format="pdf")
    dol.save(os.path.join("saved_models", f"{test}", f"{wave}_model_{th}"))

time_horizon = np.array(time_horizon)
maes = np.array(maes).flatten()
print(f"tau = {time_horizon} s")
print(f"MAE = {maes} ")
fig = plt.figure(figsize=(6, 6))
plt.bar(time_horizon, maes)
plt.xlabel('prediction horizon (s)')
plt.ylabel('MAE')
plt.savefig(os.path.join("figures", f"{test}", f"{wave}_MAE_vs_time_horizon.pdf"), format="pdf")