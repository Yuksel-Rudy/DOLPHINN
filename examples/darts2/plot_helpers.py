from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

def plot_val(original, predicted, components, mask_log=None):
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
        original[comp].plot(label="actual" if i == 0 else "", ax=ax, color='blue')
        predicted[comp].plot(label="predicted" if i == 0 else "", ax=ax, color='orange')
        ax.set_title(f"component: {comp}")

        # Highlight masked regions
        if mask_log:
            for mask in mask_log:
                if comp in mask["components"]:
                    t_index = original.time_index
                    t0, t1 = t_index[mask["temporal_range"][0]], t_index[mask["temporal_range"][1]]
                    label = f"{mask['type']}={mask['value']}" if i == 0 else None
                    ax.axvspan(t0, t1, color='yellow', alpha=0.2, label=label)


    axes[0].legend()
    plt.xlabel("t (s)")
    all_vals = original[list(components)].all_values()
    ymin, ymax = all_vals.min(), all_vals.max()
    plt.ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))
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

def plot_violin_tau_distribution(taus, n, dt):
    """Plots a violin plot of normalized prediction horizons."""
    df = pd.DataFrame(taus / (n * dt))
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