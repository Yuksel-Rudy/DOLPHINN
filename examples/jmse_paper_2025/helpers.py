import numpy as np
from darts import TimeSeries
import pandas as pd
from darts import concatenate
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


def interpolateData(df, dt):
    time = df["Time"].to_numpy()
    vals = df.iloc[:, 1:].to_numpy()
    
    time_ = np.arange(time[0], time[-1] + dt, dt)
    vals_ = np.zeros((len(time_), vals.shape[1]), dtype=np.float32)
    for i in range(vals.shape[1]):
        vals_[:, i] = np.interp(time_, time, vals[:, i])

    # --- Convert to TimeSeries ---
    start_time = pd.Timestamp("2025-01-01")
    datetime_index = start_time + pd.to_timedelta(time_, unit="s")


    df_ = pd.DataFrame(vals_, index=datetime_index, columns=df.columns[1:])
    df_.index.name = "Time"
    series = TimeSeries.from_dataframe(df_)

    return series

def split_ts(ts, cov_idx, tgt_idx):
    cov_names = [ts.components[i] for i in cov_idx]
    tgt_names = [ts.components[i] for i in tgt_idx]    

    cov_series = ts[cov_names]
    tgt_series = ts[tgt_names]

    return tgt_series, cov_series


def mae_hist(original, predicted):
    e = (original - predicted)
    e = e.flatten()

    mae = np.mean(np.abs(e))
    plt.figure(figsize=(10, 6))
    plt.hist(e[e > 0], bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.hist(e[e < 0], bins=50, color='red', alpha=0.7, edgecolor='black')
    # also add a horizontal line at mean of absolute(e)
    plt.axvline(mae, color='black', linestyle='dashed', linewidth=1.5, label='MAE')
    plt.title("Histogram of Error")
    plt.xlabel("AE (m)")
    plt.ylabel("Frequency") 
    plt.legend()
    plt.grid()

    # plt.show()



def demean_series(ts):
    """
    Demeans the TimeSeries by subtracting the mean of each component.
    
    Parameters:
    - ts: TimeSeries object to be demeaned.
    
    Returns:
    - Demeaned TimeSeries object.
    """
    values = ts.values(copy=True)  # shape: [time, component]
    means = np.mean(values, axis=0)
    values -= means

    return TimeSeries.from_times_and_values(ts.time_index, values, columns=ts.components)

def uncertainty(ts, quantiles=[0.025, 0.975], sigma=3.0):
    """
    Computes the uncertainty bounds for a TimeSeries based on specified quantiles.
    
    Parameters:
    - ts: TimeSeries object.
    - quantiles: List of quantiles to compute (default is [0.025, 0.975]).
    
    Returns:
    - Tuple of lower and upper bounds as TimeSeries objects.
    """
    array = ts._xa.values
    lower_q = np.quantile(array, quantiles[0], axis=2)
    upper_q = np.quantile(array, quantiles[1], axis=2)
    median_q= np.quantile(array, 0.5, axis=2)

    delta = (upper_q - lower_q)
    delta_smooth = gaussian_filter1d(delta, sigma=sigma)
    return delta, delta_smooth

def get_predictionHorizon(ts, delta_0='solve_for', plot=False):
    """
    Computes the prediction horizon for a probabilistic TimeSeries forecast based on uncertainty bounds.

    Parameters:
    - ts: TimeSeries object containing probabilistic forecasts (with samples as the last dimension).
    - delta_0: Threshold for uncertainty to determine the prediction horizon. If 'solve_for', it is set automatically.
    - plot: If True, plots the uncertainty and prediction horizon.

    Returns:
    - tau: The minimum prediction horizon (in seconds) where uncertainty exceeds the threshold.
    - delta_0: The threshold value used for uncertainty (minimum in case of multiple targets).
    """
    if len(ts) < 2:
        raise ValueError("TimeSeries is too short.")
    
    dt = (ts.time_index[1] - ts.time_index[0]).total_seconds() 
    delta, delta_smooth = uncertainty(ts)
    delta_max = delta.max(axis=0)
    delta_min = delta.min(axis=0)
    if delta_0 == 'solve_for':
        delta_0_min = np.min(delta_min + (delta_max - delta_min) / 4.0) 
        delta_0_mean = np.mean(delta_min + (delta_max - delta_min) / 4.0) 
        delta_0_max = np.max(delta_min + (delta_max - delta_min) / 4.0) 
        delta_0 = delta_0_mean
    else:
        delta_0 = delta_0
    # Apply smoothing to delta using Gaussian Kernel smoothing
    taus = []
    for i in range(delta.shape[1]):
        cond = np.sum(delta_smooth[:, i] < delta_0)
        tau_sec = cond * dt
        taus.append(tau_sec)    
    
    if plot:
        plt.rcParams["font.family"] = "Times New Roman"
        fig, axs = plt.subplots(delta.shape[1], 1, figsize=(10, 3 * delta.shape[1]), sharex=True)
        if delta.shape[1] == 1:
            axs = [axs]
        for i, ax in enumerate(axs):
            ax.plot(ts.time_index, delta[:, i], label=r"$\delta$", alpha=0.5)
            ax.plot(ts.time_index, delta_smooth[:, i], label=r"$\delta$ Smoothed")
            ax.axhline(delta_0_max, color='red', linestyle='--', label='Threshold - max')
            ax.axhline(delta_0_mean, color='cyan', linestyle='--', label='Threshold - mean')
            ax.axhline(delta_0_min, color='magenta', linestyle='--', label='Threshold - min')
            if taus[i] > 0:
                tau_time = ts.time_index[int(taus[i] / dt)-1]
                ax.axvline(tau_time, color='green', linestyle='--', label='Prediction horizon')
            ax.set_ylabel(r'$\delta$ (Level)')
            ax.legend()
            ax.grid()
        axs[-1].set_xlabel('Time')
        fig.tight_layout()

    return taus, delta_0

def rolling_probabilistic_forecast(
    model, 
    series, 
    past_covariates, 
    input_chunk_length, 
    forecast_horizon,
    components=None,
    stride=1, 
    num_samples=1000, 
    delta_0='solve_for'
):
    forecasts_scaler   = []
    time_starts = []
    minTaus        = []
    delta_0s    = []
    strides = range(input_chunk_length, len(series) - forecast_horizon + 1, stride)
    if components is None:
        components = series.components    
    allTaus = np.zeros((len(strides), len(components)))
    for i, s in enumerate(strides):
        hist = series[:s]
        cov = past_covariates[:s] if past_covariates is not None else None
        
        if cov is not None:
            forecast_scaler = model.predict(
                n=forecast_horizon,
                series=hist,
                past_covariates=cov,
                num_samples=num_samples,
            )
        else:
            forecast_scaler = model.predict(
                n=forecast_horizon,
                series=hist,
                num_samples=num_samples,
            )
            forecasts_scaler.append(forecast_scaler)
        time_starts.append(hist.end_time())
        forecast_scaler = forecast_scaler[list(components)]
        taus, _ = get_predictionHorizon(forecast_scaler, delta_0=delta_0, plot=False)
        # plt.show()
        allTaus[i, :] = taus
        print(f"Forecasting from {hist.start_time()} to {hist.end_time()} with minTau={np.mean(taus):.2f} s")
        minTaus.append(np.mean(taus))

        # Find the most frequent tau
        if len(minTaus) > 1:
            tauCounts = np.bincount(minTaus)
            mostFrequentminTau = np.argmax(tauCounts)

    return forecasts_scaler, time_starts, minTaus, allTaus, mostFrequentminTau


def print_tau_summary(minTaus, mostFrequentminTau):
    """Prints summary statistics."""
    print(f"Mean prediction horizon: {np.mean(minTaus):.2f} s")
    print(f"Most frequent tau: {mostFrequentminTau:.2f} s")