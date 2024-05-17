import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from vmod.dolphinn import DOLPHINN as DOL
from vmod.p2v import get_psd
import numpy as np
"""
In this example:
- train MLSTM model using DOLPHINN on wavestaff data alone
"""

# Configure
TEST = "1d"
WAVE = "IR1"
CONFIG_FILE_PATH = os.path.join("..", "dol_input", f"{WAVE}.yaml")
if not os.path.exists(os.path.join("..", "figures", f"{TEST}")):
    os.makedirs(os.path.join("..", "figures", f"{TEST}"))

# call dolphinn
dol = DOL()
dol.train(config_path=CONFIG_FILE_PATH, labels_to_be_dropped=[1, 2, 3, 4])
r_square, mae, y, y_hat = dol.test()
t = np.linspace(0, (y.shape[0]-1)*dol.timestep, y.shape[0])

plt.rcParams.update({'font.size': 18})
matplotlib.rcParams['font.family'] = 'Times New Roman'

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
    ax.set_title(f"{WAVE}")

plt.tight_layout()
plt.savefig(os.path.join("..", "figures", f"{TEST}", f"{WAVE}_TD.pdf"), format="pdf")

# post-processing (FD)
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(len(dol.labels), 1)

for i, (label, unit) in enumerate(zip(dol.labels, dol.unit)):
    F, PSD_exp = get_psd.get_PSD_limited(t, y[:, i] - np.mean(y[:, i]), 30, 0, 0.5)
    _, PSD_wrp = get_psd.get_PSD_limited(t, y_hat[:, i] - np.mean(y_hat[:, i]), 30, 0, 0.5)
    ax = plt.subplot(gs[i])
    ax.plot(F, PSD_exp, label='experiment', color='black')
    ax.plot(F, PSD_wrp, label='DOLPHINN', color='red')
    ax.set_xlabel('f [Hz]')
    ax.set_ylabel(rf"wave elevation PSD $[{unit[1:-1]}^2/Hz]$")
    ax.set_xlim((0, 0.25))
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_title(f"{WAVE}")

plt.tight_layout()
plt.savefig(os.path.join("..", "figures", f"{TEST}", f"{WAVE}_FD.pdf"), format="pdf")

# save dolphinn
dol.save(os.path.join("..", "saved_models", f"{TEST}", f"{WAVE}_model"))