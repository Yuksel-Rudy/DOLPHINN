import os
import pandas as pd
import numpy as np
from vmod.dolphinn import DOLPHINN as DOL
from vmod.p2v import zero_crossing as zc
def run_prediction(
    modelDir,
    tablePath,
    deltaTime=0,
    history=1000,
    numShots=1,
    plotResult=True,
    targetColumn="wave5"
):
    # --- Load model and table ---
    dol = DOL()
    dol.load(modelDir)
    table = pd.read_csv(tablePath)

    # --- Find the time where the highest wave occured
    # Apply zero-crossing on the most downstream probe
    T, H, sT, sT_idx = zc(table['Time'].values, table[dol.dof[-1]].values)
    # Find the index of the maximum wave height
    max_idx = np.argmax(H)

    # --- Determine time window indices ---
    t_present = sT[max_idx] + deltaTime  # Time where highest wave is observed on first probe
    t_horizon = dol.time_horizon
    t_future = t_present + numShots * t_horizon  # For the first iteration

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
    if plotResult and (targetColumn in data.columns or targetColumn == "ALL"):
        import matplotlib.pyplot as plt
        if targetColumn in data.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(table['Time'].iloc[:present_idx], table[targetColumn].iloc[:present_idx], label='Observed Past', color='black')
            plt.plot(table['Time'].iloc[present_idx:future_idx], table[targetColumn].iloc[present_idx:future_idx], label='Actual Future', color='green')
            plt.plot(t_pred, y_pred[targetColumn], label='Predicted', linestyle='--', color='red')
            plt.xlabel("Time")
            plt.ylabel(targetColumn)
            plt.title(f"Prediction for {targetColumn}")
            plt.xlim(t_present - 50, t_present + numShots * t_horizon)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # --- Calculate Mean Absolute Error (MAE) ---
            # interpolate table to match prediction time
            actual = np.interp(t_pred, table['Time'].iloc[present_idx:future_idx], table[targetColumn].iloc[present_idx:future_idx])
            predicted = y_pred[targetColumn].values
            mae = np.abs(predicted - actual)
            plt.figure(figsize=(10, 5))
            plt.plot(t_pred, mae, label='MAE', color='gray', alpha=0.4)
            mae_smooth = pd.Series(mae).rolling(window=20, center=True).mean()
            plt.plot(t_pred, mae_smooth, label='Smoothed MAE', color='orange', linewidth=2)
            plt.xlabel("Time")
            plt.ylabel("MAE")
            plt.title("Mean Absolute Error (MAE) between Targeted and Predicted")
            plt.grid(True)
            plt.legend()
            plt.show()
        elif plotResult and targetColumn == "ALL":
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
            # Plot MAE for all columns
            fig, axes = plt.subplots(num_columns, 1, figsize=(10, 4 * num_columns), sharex=True)
            for i, col in enumerate(data.columns):
                ax = axes[i] if num_columns > 1 else axes
                # Interpolate actual values
                actual = np.interp(t_pred, table['Time'].iloc[present_idx:future_idx], table[col].iloc[present_idx:future_idx])
                predicted = y_pred[col].values
                mae = np.abs(predicted - actual)
                ax.plot(t_pred, mae, label=f"MAE: {col}", color='blue')
                ax.set_ylabel("MAE")
                ax.grid(True)
                ax.legend()

            axes[-1].set_xlabel("Time") if num_columns > 1 else axes.set_xlabel("Time")
            fig.suptitle("Mean Absolute Error (MAE) for All Columns")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()            

    return t_pred, y_pred

# --- Script execution ---
if __name__ == "__main__":
    # --- Configuration ---
    trial_name = "trial"
    input_name = "5v_mimo_th20_nm10"
    modelDir = os.path.join("saved_models", trial_name, f"model_{input_name}")
    tablePath = os.path.join("data", "FOCAL_wavedata", "scaledup", "IR-1.csv")

    # --- Run ---
    run_prediction(
        modelDir=modelDir,
        tablePath=tablePath,
        deltaTime=-40,
        history=1000,
        numShots=3,
        plotResult=True,
        targetColumn="wave5"
    )
