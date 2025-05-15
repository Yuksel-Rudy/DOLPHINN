import os
import pandas as pd
import numpy as np
from vmod.dolphinn import DOLPHINN as DOL

def run_prediction(
    model_dir,
    table_path,
    start_time=100,
    present_time=8000,
    history=1000,
    numShots=1,
    plot_result=True,
    target_column="wave5"
):
    # --- Load model and table ---
    dol = DOL()
    dol.load(model_dir)
    table = pd.read_csv(table_path)

    # --- Determine time window indices ---
    t_start = start_time
    t_present = present_time
    t_horizon = dol.time_horizon
    t_future = t_present + numShots * t_horizon  # For the first iteration

    start_idx   = np.argmin(np.abs(table['Time'] - t_start))
    present_idx = np.argmin(np.abs(table['Time'] - t_present))
    future_idx  = np.argmin(np.abs(table['Time'] - t_future))

    # --- Extract data slices ---
    time = table['Time'].iloc[:present_idx]
    data = table[dol.dof].mul(dol.conversion, axis=1).iloc[:present_idx]

    # --- Predict ---
    t_pred, y_pred = dol.predict(
        time=time,
        data=data,
        history=history,
        numShots=numShots
    )

    # --- Optional: Custom Overlay Plot ---
    if plot_result and (target_column in data.columns or target_column == "ALL"):
        import matplotlib.pyplot as plt
        if target_column in data.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(table['Time'].iloc[:present_idx], table[target_column].iloc[:present_idx], label='Observed Past', color='black')
            plt.plot(table['Time'].iloc[present_idx:future_idx], table[target_column].iloc[present_idx:future_idx], label='Actual Future', color='green')
            plt.plot(t_pred, y_pred[target_column], label='Predicted', linestyle='--', color='red')
            plt.xlabel("Time")
            plt.ylabel(target_column)
            plt.title(f"Prediction for {target_column}")
            plt.xlim(t_present - 50, t_present + numShots * t_horizon)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        elif plot_result and target_column == "ALL":
            num_columns = len(data.columns)
            fig, axes = plt.subplots(num_columns, 1, figsize=(10, 5 * num_columns), sharex=True)
            for i, col in enumerate(data.columns):
                ax = axes[i] if num_columns > 1 else axes
                ax.plot(table['Time'].iloc[:present_idx], table[col].iloc[:present_idx], label='Observed Past', color='black')
                ax.plot(table['Time'].iloc[present_idx:future_idx], table[col].iloc[present_idx:future_idx], label='Actual Future', color='green')
                ax.plot(t_pred, y_pred[col], label='Predicted', linestyle='--', color='red')
                ax.set_ylabel(col)
                ax.set_xlim(t_present - 50, t_present + numShots * t_horizon)
                ax.grid(True)
            ax.legend()
            axes[-1].set_xlabel("Time") if num_columns > 1 else axes.set_xlabel("Time")
            fig.suptitle("Prediction for All Columns")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

    return t_pred, y_pred

# --- Script execution ---
if __name__ == "__main__":
    # --- Configuration ---
    trial_name = "trial"
    input_name = "5v_mimo"
    model_path = os.path.join("saved_models", trial_name, f"model_{input_name}")
    table_path = os.path.join("data", "FOCAL_wavedata", "scaledup", "IR-1.csv")

    # --- Run ---
    run_prediction(
        model_dir=model_path,
        table_path=table_path,
        start_time=100,
        present_time=8000,
        history=1000,
        numShots=1,
        plot_result=True,
        target_column="ALL"
    )
