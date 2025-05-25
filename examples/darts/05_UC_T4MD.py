import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def maskFunction(
    data_array,
    cov_idx,
    tgt_idx,
    group_indices,
    group_to_idx,
    threshold=0.2,
    window_size=200,  # e.g., if dt = 0.5s → 100s
    seed=None,
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

    # Concatenate switches to covariates
    masked_cov_with_switch = np.concatenate([masked_cov, switch_array], axis=1)

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

# Upstream as covariates: columns 0–lastCovariateIndex
lastCovariateIndex = 20
df_covariates = pd.DataFrame(prob_interp[:, :lastCovariateIndex], index=datetime_index)
df_covariates.index.name = "Time"
past_covariates = TimeSeries.from_dataframe(df_covariates)

# Downstream as target: columns lastCovariateIndex+
df_target = pd.DataFrame(prob_interp[:, lastCovariateIndex+1:], index=datetime_index)
df_target.index.name = "Time"
target_series = TimeSeries.from_dataframe(df_target)

# Indices for covariates and targets
cov_idx = list(range(0, lastCovariateIndex))   # upstream probes
tgt_idx = list(range(lastCovariateIndex, 24))  # downstream probes


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

# Define Probe groupings

# Total 24 probes (0–23): 0–20 upstream (cov), 21–23 downstream (target)
group_labels = ['Lv1', 'Lv2', 'Lv3', 'Lv4']
group_indices = {
    group_labels[0]: list(range(0, 6)),
    group_labels[1]: list(range(6, 12)),
    group_labels[2]: list(range(12, 18)),
    group_labels[3]: list(range(18, 24))
}
group_to_idx = {name: i for i, name in enumerate(group_labels)}


from darts.models import TCNModel
from darts import TimeSeries

# --- Settings ---
m = 30
n = 10
ein = 100
eex = 10
threshold = 0.1


time_idx = full_target_scaled.time_index

# --- Create TCN model ---
model = TCNModel(
    input_chunk_length=m,
    output_chunk_length=n,
    n_epochs=ein,
    kernel_size=3,
    num_filters=3,
    num_layers=4,
    dropout=0.1,
    weight_norm=True,
    random_state=41
)

# Stack all channels for joint masking (shape: [T, 24, 1])
data_array = full_target_scaled.stack(full_cov_scaled).all_values(copy=True)

# ---- Training Loop ----
for epoch in range(eex):
    # Generate randomized masked input
    masked_cov_array, masked_tgt_array = maskFunction(
        data_array=data_array,
        cov_idx=cov_idx,
        tgt_idx=tgt_idx,
        group_indices=group_indices,
        group_to_idx=group_to_idx,
        threshold=threshold,
        window_size=400,  # e.g., if dt = 0.5s → 100s
        seed=None,  # random every time
        plot=False
    )

    # Convert to TimeSeries
    masked_cov_ts = TimeSeries.from_times_and_values(time_idx, masked_cov_array)
    masked_tgt_ts = TimeSeries.from_times_and_values(time_idx, masked_tgt_array)

    # Train for one epoch
    model.fit(series=masked_tgt_ts, past_covariates=masked_cov_ts, verbose=True)

    print(f"Epoch {epoch + 1} completed.")


# --- Save the model ---
modelName = f"tcn_C21_T4MD_m_{m}_n{n}_ein{ein}_eex{eex}" 
modelPath = os.path.join(filePath, modelName)
model.save(modelPath, "model")