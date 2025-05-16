import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import os
import sys
from pathlib import Path
from vmod.dolphinn import DOLPHINN as DOL
from vmod.p2v import get_psd
import numpy as np

plt.rcParams.update({'font.size': 18})
matplotlib.rcParams['font.family'] = 'Times New Roman'

trial = "trial"
input = "5v_mimo_th20_nm10"

config_file_path = os.path.join("dol_input", f"{trial}", f"{input}.yaml")
dol = DOL(config_path=config_file_path)
dol.train()
r_sq, mae, y, y_hat = dol.test()
t = np.linspace(0, (y.shape[0]-1)*dol.timestep, y.shape[0])
t_hat = t - dol.time_horizon

print(f"R^2: {r_sq}")
print(f"MAE: {mae}")
fig, ax = plt.subplots()
ax.plot(t, y[:, -1], label='experiment', color='black')
ax.plot(t, y_hat[:, -1], label='DOLPHINN', color='red', linestyle='-')
plt.show()

# Save the model
model_dir = os.path.join("saved_models", f"{trial}", f"model_{input}")
print(f'Saving model to {model_dir}')
dol.save(model_dir)