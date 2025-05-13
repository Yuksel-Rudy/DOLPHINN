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
time_horizon = 80  # in seconds

DOLPHINN_PATH = os.path.join("saved_models", f"{test}", f"{wave}_model_{time_horizon}")
if not os.path.exists(os.path.join("figures", f"{test}")):
    os.makedirs(os.path.join("figures", f"{test}"))

# call dolphinn
dol = DOL()
dol.load(DOLPHINN_PATH)
r_square, mae, y, y_hat = dol.test()

# post-processing (TD)
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(len(dol.labels), 1)
for i, (label, unit) in enumerate(zip(dol.labels, dol.unit)):    
    # Calculate the range of y and y_hat
    y_range = np.max(y[:, i]) - np.min(y[:, i])
    y_hat_range = np.max(y_hat[:, i]) - np.min(y_hat[:, i])
    relative_size = y_hat_range / y_range

print(f"Relative Amplitude: {relative_size:.2f}")