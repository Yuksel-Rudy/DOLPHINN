from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

plt.rcParams["font.family"] = "Times New Roman"

def plot_error(original, predicted, sigma=5, components=None):
    """
    Plot smoothed time-varying MAE between tgt_val and backtest using a Gaussian filter.
    
    Parameters:
    - tgt_val: ground truth TimeSeries
    - backtest: predicted TimeSeries
    - sigma: standard deviation for Gaussian filter (smoothing strength)
    - components: list of component names to include (default: all target components)
    """
    if components is None:
        components = original.components

    time_index = predicted.time_index
    fig, ax = plt.subplots(nrows=len(components), ncols=1, figsize=(12, 4 * len(components)), sharex=True)

    if len(components) == 1:
        ax = [ax]

    for i, comp in enumerate(components):
        pred_vals = predicted[comp].values().flatten()
        true_vals = original[comp].values().flatten()[-len(pred_vals):]
        abs_error = np.abs(true_vals - pred_vals)
        smoothed_error = gaussian_filter1d(abs_error, sigma=sigma)

        ax[i].plot(time_index, abs_error, label=f"MAE: {comp}", color="blue", alpha=0.25)
        ax[i].plot(time_index, smoothed_error, label=f"Gaussian filtered MAE: {comp}", color="crimson")
        ax[i].set_ylabel("MAE (m)")
        ax[i].legend()
        ax[i].grid(True, linestyle=":", linewidth=0.5)

    ax[-1].set_xlabel("t (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
def plot_val(original, predicted, components, other_series=None, mask_log=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    if isinstance(components, str):
        components = [components]

    fig, axes = plt.subplots(nrows=len(components), ncols=1, figsize=(10, 6), sharex=True)

    if len(components) == 1:
        axes = [axes]

    for i, comp in enumerate(components):
        ax = axes[i]
        original[comp].plot(ax=ax, color='blue')
        predicted[comp].plot(ax=ax, color='orange')
        if other_series is not None:    
            other_series[comp].plot(ax=ax, color='black')
        # Highlight masked regions
        if mask_log:
            for mask in mask_log:
                if comp in mask["components"]:
                    t_index = original.time_index
                    t0, t1 = t_index[mask["temporal_range"][0]], t_index[mask["temporal_range"][1]]
                    label = f"{mask['type']}={mask['value']}" if i == 0 else None
                    ax.axvspan(t0, t1, color='yellow', alpha=0.2, label=label)
                    all_vals = original[list(components)].all_values()
                    ymin, ymax = all_vals.min(), all_vals.max()                    
                    ax.set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))
                    ax.set_xlim(original.time_index[500], original.time_index[2000])

    plt.xlabel("")  # Remove x-label
    plt.ylabel("")  # Remove y-label
    plt.xlabel("t (s)")
    plt.tight_layout()



def plot_prediction_histogram(taus):
    """Plots a histogram of prediction horizons."""
    plt.figure(figsize=(8, 5))
    plt.hist(taus, bins=30, edgecolor='k', alpha=0.7)
    plt.axvline(np.mean(taus), color='r', linestyle='dashed', linewidth=1, label='mean')
    plt.legend()
    plt.xlabel(r'prediction horizon, $\tau$ (s)')
    plt.ylabel('frequency')
    plt.title('histogram of prediction horizons')
    plt.tight_layout()

def plot_violin_tau_distribution(taus, n, dt, components=None):
    """Plots a violin plot of normalized prediction horizons."""
    plt.rcParams.update({'font.size': 14})
    if components is None:
        components = range(taus.shape[1])  # Use all components if none specified 
    df = pd.DataFrame(taus / (n * dt), columns=components)
    df_melt = df.melt(var_name="component", value_name=r"$\tau / \tau_{max}$")
    plt.figure(figsize=(10, 4))
    sns.violinplot(data=df_melt, x="component", y=r"$\tau / \tau_{max}$", inner="quartile")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.title("distribution of normalized prediction horizons")
    plt.tight_layout()
    

def plot_tau_heatmap(taus, n, dt):
    """Plots a heatmap of tau densities across components."""
    taus_norm = taus / (n * dt)

    hist_data, xedges, yedges = np.histogram2d(
        np.tile(np.arange(taus_norm.shape[1]), taus_norm.shape[0]),
        taus_norm.flatten(),
        bins=[taus_norm.shape[1], 50],
        range=[[0, taus_norm.shape[1]], [0, 1]]
    )

    plt.figure(figsize=(10, 4))
    plt.imshow(hist_data.T, aspect='auto', origin='lower', cmap='viridis',
               extent=[0, taus_norm.shape[1], 0, 1])
    plt.colorbar(label='frequency')
    plt.xlabel('component index')
    plt.ylabel(r'$\tau/\tau_{max}$')
    plt.title("heatmap of prediction horizon distribution")
    plt.tight_layout()