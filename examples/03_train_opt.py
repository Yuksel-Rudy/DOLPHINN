import optuna
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from vmod.dolphinn import DOLPHINN as DOL

"""
In this example:
- Optimize hyper-parameters to train MLSTM-WRP on wind/wave data
"""

def objective(trial):
    print(f"performing trial {trial.number}")

    # Hyperparameters to be optimized
    nm = trial.suggest_float("n/m", 0.25, 1.0)
    hidden_layer = trial.suggest_int("hidden_layer", 1, 4)
    neuron_number = trial.suggest_int('total number of neurons', 25, 100)
    epochs = trial.suggest_int('epochs', 40, 200)
    batch_time = trial.suggest_int('batch time', 40, 80)
    timestep = trial.suggest_float('timestep', 0.25, 1.0)
    lr = trial.suggest_float('learning rate', 0.0001, 0.1)
    dropout = trial.suggest_float('dropout', 0.0, 0.8)
    # Train_Test ratio
    train_ratio = trial.suggest_float('train ratio', 0.5, 0.74)
    valid_ratio = 1 - 0.25 - train_ratio

    # Applying hyperparameters
    dol.nm = nm
    dol.hidden_layer = hidden_layer
    dol.neuron_number = neuron_number
    dol.epochs = epochs
    dol.batch_time = batch_time
    dol.timestep = timestep
    dol.lr = lr
    dol.dropout = dropout
    dol.train_ratio = train_ratio
    dol.valid_ratio = valid_ratio
    dol.train()
    r_square, mae, y, y_hat = dol.test()
    obj = mae.mean()

    return obj


TEST = "3"
CONFIG_FILE_PATH = os.path.join("dol_input", "init_dol.yaml")
if not os.path.exists(os.path.join("figures", f"{TEST}")):
    os.makedirs(os.path.join("figures", f"{TEST}"))

# call dolphinn
dol = DOL()
# Load configuration
dol.config_path = CONFIG_FILE_PATH
dol.load_config()

study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=50, multivariate=True, group=True),
                            direction='minimize')
study.optimize(objective, n_trials=1000, gc_after_trial=True)

# Applying optimized hyperparameters
dol.nm = study.best_params["n/m"]
dol.hidden_layer = study.best_params["hidden_layer"]
dol.neuron_number = study.best_params["total number of neurons"]
dol.epochs = study.best_params["epochs"]
dol.batch_time = study.best_params["batch time"]
dol.timestep = study.best_params["timestep"]
dol.lr = study.best_params["learning rate"]
dol.dropout = study.best_params["dropout"]
dol.train_ratio = study.best_params["train ratio"]
dol.valid_ratio = 1 - 0.25 - dol.train_ratio
dol.train()
r_square, mae, y, y_hat = dol.test()

# save dolphinn
dol.save(os.path.join("saved_models", f"{TEST}", "OPT_model"))