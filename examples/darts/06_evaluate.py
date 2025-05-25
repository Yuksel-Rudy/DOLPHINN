import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae

def maskFunction(
    data_array,
    cov_idx,
    tgt_idx,
    group_indices,
    group_to_idx,
    threshold=0.2,
    window_size=200,  # e.g., if dt = 0.5s → 100s
    seed=None,
    switch=False,
    plot=False
):
    if seed is not None:
        np.random.seed(seed)

    data_array_md = data_array.copy()
    n_time = data_array.shape[0]
    n_cov = len(cov_idx)
    n_tgt = len(tgt_idx)
    n_groups = 4
    masked_cov = data_array[:, cov_idx, :].copy()
    masked_tgt = data_array[:, tgt_idx, :].copy()
    switch_array = np.ones((n_time, n_groups, 1), dtype=np.float32)

    # Apply random masking in each time window
    for start in range(0, n_time, window_size):
        end = min(start + window_size, n_time)

        group_mask_counts = {k: 0 for k in group_indices}
        mask_decision = np.random.rand(data_array.shape[1]) < threshold

        for ch in range(data_array.shape[1]):
            if mask_decision[ch]:
                if ch in cov_idx:
                    masked_cov[start:end, cov_idx.index(ch), :] = 0.0
                if ch in tgt_idx:
                    masked_tgt[start:end, tgt_idx.index(ch), :] = 0.0
                
                data_array_md[start:end, ch, :] = 0.0
                for group_name, ch_idx in group_indices.items():
                    if ch in ch_idx:
                        group_mask_counts[group_name] += 1
                        break

        # Compute group switch values for this window
        for group_name, masked_count in group_mask_counts.items():
            active_ratio = 1.0 - masked_count / len(group_indices[group_name])
            switch_array[start:end, group_to_idx[group_name], :] = active_ratio

    if switch:
        # Concatenate switches to covariates
        masked_cov_with_switch = np.concatenate([masked_cov, switch_array], axis=1)
    else:
        masked_cov_with_switch = masked_cov.copy()
    # Optional Plot
    if plot:
        import matplotlib.pyplot as plt
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        fig, ax = plt.subplots(figsize=(12, 6))
        time = np.arange(n_time)

        # Plot all probe channels (by group)
        for group_name, color in zip(group_indices.keys(), colors):
            for ch in group_indices[group_name]:
                if ch in cov_idx:
                    ax.plot(time, data_array_md[:, ch, 0], color=color, alpha=0.3, lw=1)

        # Plot switches (bold lines)
        for i, (group_name, color) in enumerate(zip(group_indices.keys(), colors)):
            ax.plot(time, switch_array[:, i, 0], color=color, lw=3, label=f"{group_name} switch")

        ax.set_title("Windowed Probe Masking and Switches")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Amplitude / Switch")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return masked_cov_with_switch, masked_tgt

def temporal_masking_analysis(model, val_cov_scaled, val_target_scaled,
                               cov_idx, tgt_idx, group_indices, group_to_idx, n,
                               maskFunction, switch=False, window_size=200, threshold=0.1, seed=None, ):
    """
    Applies randomized temporal masking and visualizes:
      - Rolling MAE over time
      - Dropout count per probe group as a stacked bar (background)
    """
    time_idx = val_cov_scaled.time_index
    data_array = val_cov_scaled.stack(val_target_scaled).all_values(copy=True)

    # Run masking
    masked_cov_array, masked_tgt_array = maskFunction(
        data_array=data_array,
        cov_idx=cov_idx,
        tgt_idx=tgt_idx,
        group_indices=group_indices,
        group_to_idx=group_to_idx,
        threshold=threshold,
        window_size=window_size,
        seed=seed,
        switch=switch,
        plot=False
    )

    # Convert to TimeSeries
    masked_cov_ts = TimeSeries.from_times_and_values(time_idx, masked_cov_array)
    masked_tgt_ts = TimeSeries.from_times_and_values(time_idx, masked_tgt_array)

    # Forecast
    backtest = model.historical_forecasts(
        series=masked_tgt_ts,
        past_covariates=masked_cov_ts,
        forecast_horizon=n,
        retrain=False,
        verbose=False
    )

    # Rolling MAE
    step = window_size
    time_points = np.arange(0, len(time_idx) - step, step)
    rolling_mae = []
    for start in time_points:
        end = start + step
        error = np.mean([
            mae(val_target_scaled[comp].slice(start, end), backtest[comp].slice(start, end))
            for comp in val_target_scaled.components[-4:]
        ])
        rolling_mae.append(error)

    # Compute dropout per group (in switch channels)
    switch_array = masked_cov_array[:, -4:, 0]
    group_sizes = np.array([len(group_indices[g]) for g in group_indices])
    masked_counts = (1.0 - switch_array) * group_sizes

    group_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(time_points, rolling_mae, color='black', label='Rolling MAE')
    ax1.set_ylabel("MAE")
    ax1.set_xlabel("Time Step")
    ax1.set_title("MAE with Temporal Dropout Masking")
    ax1.grid(True)

    ax2 = ax1.twinx()
    bottom = np.zeros_like(time_points, dtype=float)

    for i, group_name in enumerate(group_indices):
        group_mask_avg = [
            np.mean(masked_counts[start:start + step, i])
            for start in time_points
        ]
        ax2.bar(time_points, group_mask_avg, bottom=bottom, width=step, color=group_colors[i],
                alpha=0.3, label=f"{group_name}")
        bottom += group_mask_avg

    ax2.set_ylabel("Average Masked Probes per Group")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def offline_probe_sweep_analysis(model, val_cov_scaled, val_target_scaled, n, max_offline=21, switch=False):
    """
    Evaluates model performance (MAE) while increasing number of offline upstream probes.
    """
    mae_list = []
    offline_counts = list(range(0, max_offline + 1))
    time_idx = val_cov_scaled.time_index

    for k in offline_counts:
        idx_cov = list(range(0, k))  # progressively offline the first k covariates
        masked_cov_array = val_cov_scaled.all_values(copy=True)
        masked_tgt_array = val_target_scaled.all_values(copy=True)
        masked_cov_array[:, idx_cov, :] = 0.0

        if switch:
            # dummy switch array:
            n_time = masked_cov_array.shape[0]
            switch_array = np.ones((n_time, 4, 1), dtype=np.float32)
            masked_cov_array = np.concatenate([masked_cov_array, switch_array], axis=1)

        masked_cov_ts = TimeSeries.from_times_and_values(time_idx, masked_cov_array)
        masked_tgt_ts = TimeSeries.from_times_and_values(time_idx, masked_tgt_array)

        backtest = model.historical_forecasts(
            series=masked_tgt_ts,
            past_covariates=masked_cov_ts,
            forecast_horizon=n,
            retrain=False,
            verbose=False
        )

        # Compute average MAE over all 4 downstream probes
        mae_val = np.mean([
            mae(val_target_scaled[comp], backtest[comp])
            for comp in val_target_scaled.components[-4:]
        ])
        mae_list.append(mae_val)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(offline_counts, mae_list, color='steelblue')
    plt.xlabel("Number of Offline Upstream Probes")
    plt.ylabel("MAE over Downstream Targets")
    plt.title("Model Performance vs. Upstream Probe Failure")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return offline_counts, mae_list
# --- Load and interpolate ---
filePath = os.path.dirname(os.path.abspath(__file__))   
dataPath = os.path.join(filePath, "waveData.csv")
# dataPath = "waveData.csv"

waveData = pd.read_csv(dataPath)
time = waveData["Time"].to_numpy()
prob = waveData.iloc[:, 1:].to_numpy()

time = np.arange(time[0], time[-1], 0.5)
prob_interp = np.zeros((len(time), prob.shape[1]), dtype=np.float32)
for i in range(prob.shape[1]):
    prob_interp[:, i] = np.interp(time, waveData["Time"], prob[:, i])

# --- Convert to TimeSeries ---
start_time = pd.Timestamp("2025-01-01")
datetime_index = start_time + pd.to_timedelta(time, unit="s")

# Upstream as covariates: columns 0–20
df_covariates = pd.DataFrame(prob_interp[:, :21], index=datetime_index)
df_covariates.index.name = "Time"
past_covariates = TimeSeries.from_dataframe(df_covariates)

# Downstream as target: columns 21+
df_target = pd.DataFrame(prob_interp[:, 21:], index=datetime_index)
df_target.index.name = "Time"
target_series = TimeSeries.from_dataframe(df_target)

# --- Train/Validation split ---
train_target, val_target = target_series.split_after(0.75)
train_cov, val_cov = past_covariates.split_after(0.75)

# --- Scale data ---
scaler_target = Scaler()
scaler_cov = Scaler()

train_target_scaled = scaler_target.fit_transform(train_target)
val_target_scaled = scaler_target.transform(val_target)
full_target_scaled = scaler_target.transform(target_series)

train_cov_scaled = scaler_cov.fit_transform(train_cov)
val_cov_scaled = scaler_cov.transform(val_cov)
full_cov_scaled = scaler_cov.transform(past_covariates)


# MODEL INFORMATION
m = 240
n = 80
e = 100

from darts.models import TCNModel
modelName = os.path.join(filePath, f"tcn_C21_m_{m}_n{n}_e{e}")
model = TCNModel.load(modelName)
offline_probe_sweep_analysis(model, val_cov_scaled, val_target_scaled, n, max_offline=21, switch=False)

# Temporal masking analysis
lastCovariateIndex = 20
cov_idx = list(range(0, lastCovariateIndex))   # upstream probes
tgt_idx = list(range(lastCovariateIndex, 24))  # downstream probes4
group_labels = ['Lv1', 'Lv2', 'Lv3', 'Lv4']
group_indices = {
    group_labels[0]: list(range(0, 6)),
    group_labels[1]: list(range(6, 12)),
    group_labels[2]: list(range(12, 18)),
    group_labels[3]: list(range(18, 24))
}
group_to_idx = {name: i for i, name in enumerate(group_labels)}

temporal_masking_analysis(model, val_cov_scaled, val_target_scaled,
                          cov_idx=cov_idx, tgt_idx=tgt_idxtgt_idx,
                          group_indices=group_indices,
                          group_to_idx=group_to_idx,
                          n=n, maskFunction=maskFunction, switch=False, window_size=200, threshold=0.1, seed=42)

