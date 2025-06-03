from darts import TimeSeries
import numpy as np
from vmod.zero_crossing import zero_up_crossing

class nwtTiDEMasks():
    def __init__(self, nComponents, nDivs, cov, tgt, maskFactor=0.0):
        self.nComponents = nComponents
        self.nDivs = nDivs
        self.cov = cov
        self.tgt = tgt
        self.maskFactor = maskFactor

        # Temporal masking
        ind0 = np.random.randint(0, self.cov.shape[0])
        ind1 = np.random.randint(0, self.cov.shape[0])
        self.ind = [0, -1]
        self.ind[0] = min(ind0, ind1)
        self.ind[1] = max(ind0, ind1)
        
        self.affectedComp = []
    def maskA(self):
        lim0 = 0
        lim1 = 1
        maskedComp = range(int(self.nComponents/self.nDivs) * lim0, int(self.nComponents/self.nDivs) * lim1)
        self.cov = mask_ts(self.cov, components=list(self.cov.components[maskedComp]), mask_factor=self.maskFactor, ind=self.ind)
        self.affectedComp.append(list(self.cov.components[maskedComp]))
    def maskB(self):
        lim0 = 1
        lim1 = 2
        maskedComp = range(int(self.nComponents/self.nDivs) * lim0, int(self.nComponents/self.nDivs) * lim1)
        self.cov = mask_ts(self.cov, components=list(self.cov.components[maskedComp]), mask_factor=self.maskFactor, ind=self.ind)
        self.affectedComp.append(list(self.cov.components[maskedComp]))
    def maskC(self):    
        lim0 = 2
        lim1 = 3
        maskedComp = range(int(self.nComponents/self.nDivs) * lim0, int(self.nComponents/self.nDivs) * lim1)
        self.cov = mask_ts(self.cov, components=list(self.cov.components[maskedComp]), mask_factor=self.maskFactor, ind=self.ind)
        self.affectedComp.append(list(self.cov.components[maskedComp]))
    def maskD(self):
        lim0 = 0
        lim1 = 1
        maskedComp = range(int(self.nComponents/self.nDivs) * lim0, int(self.nComponents/self.nDivs) * lim1)
        self.tgt = mask_ts(self.tgt, components=list(self.tgt.components[maskedComp]), mask_factor=self.maskFactor, ind=self.ind)
        self.affectedComp.append(list(self.tgt.components[maskedComp]))

    def maskPhaseConstant(self, phaseStep=5):
        """
        Applies a phase shift to all target components by a constant number of steps.
        """
        arr_tgt = self.tgt.all_values(copy=True)
        arr_cov = self.cov.all_values(copy=True)

        if phaseStep == 0:
            return

        shifted_tgt = np.zeros_like(arr_tgt)
        shifted_cov = np.zeros_like(arr_cov)

        if phaseStep > 0:
            # Delay
            shifted_tgt[phaseStep:, :] = arr_tgt[:-phaseStep, :]
            shifted_cov[phaseStep:, :] = arr_cov[:-phaseStep, :]
        else:
            # Advance
            shifted_tgt[:phaseStep, :] = arr_tgt[-phaseStep:, :]
            shifted_cov[:phaseStep, :] = arr_cov[-phaseStep:, :]
        
        self.tgt = TimeSeries.from_times_and_values(
            self.tgt.time_index, shifted_tgt, columns=self.tgt.components
        )
        self.cov = TimeSeries.from_times_and_values(
            self.cov.time_index, shifted_cov, columns=self.cov.components
        )


class nwtLSTMMasks():
    def __init__(self, nComponents, nDivs, tgt, maskFactor=0.0):
        self.nComponents = nComponents
        self.nDivs = nDivs
        self.tgt = tgt
        self.maskFactor = maskFactor
        self.maskLog = []  # list of dicts, one per masking action

    def getRandomTemporalMasking(self):
        ind0 = np.random.randint(0, self.tgt.shape[0])
        ind1 = np.random.randint(0, self.tgt.shape[0])
        return [min(ind0, ind1), max(ind0, ind1)]

    def apply_mask(self, lim0, lim1, mask_type='factor', mask_value=None, randomTemporalMasking=True):
        ind = self.getRandomTemporalMasking() if randomTemporalMasking else [0, self.tgt.shape[0] - 1]
        maskedComp = range(int(self.nComponents/self.nDivs) * lim0,
                           int(self.nComponents/self.nDivs) * lim1)
        comp_list = list(self.tgt.components[maskedComp])

        self.tgt = mask_ts(self.tgt, components=comp_list, mask_factor=self.maskFactor, ind=ind)

        self.maskLog.append({
            "components": comp_list,
            "type": mask_type,
            "value": self.maskFactor if mask_value is None else mask_value,
            "temporal_range": ind
        })

    def maskA(self, randomTemporalMasking=True):
        self.apply_mask(0, 1, randomTemporalMasking=randomTemporalMasking)

    def maskB(self, randomTemporalMasking=True):
        self.apply_mask(1, 2, randomTemporalMasking=randomTemporalMasking)

    def maskC(self, randomTemporalMasking=True):
        self.apply_mask(2, 3, randomTemporalMasking=randomTemporalMasking)

    def maskD(self, randomTemporalMasking=True):
        self.apply_mask(3, 4, randomTemporalMasking=randomTemporalMasking)
    
    def maskAll(self, threshold=0.2, window_size=200, mask_type="factor"):
        n_time = self.tgt.n_timesteps
        components = list(self.tgt.components)

        for start in range(0, n_time, window_size):
            end = min(start + window_size, n_time - 1)

            for comp in components:
                if np.random.rand() < threshold:
                    # Apply masking
                    if mask_type == "factor":
                        self.tgt = mask_ts(
                            self.tgt,
                            components=[comp],
                            mask_factor=self.maskFactor,
                            ind=[start, end]
                        )
                        value = self.maskFactor
                    elif mask_type == "substitute":
                        g = 9.81  # gravitational acceleration (m/s^2)
                        T, H = self.get_waveProperties(self.tgt[comp], start, history=100)
                        if np.isnan(T) or np.isnan(H) or T <= 0:
                            self.tgt = mask_ts(
                                self.tgt,
                                components=[comp],
                                mask_factor=self.maskFactor,
                                ind=[start, end]
                            )
                            value = self.maskFactor
                            self.maskLog.append({
                                "components": [comp],
                                "type": "factor",  # forced factoring
                                "value": value,
                                "temporal_range": [start, end]
                            })                            
                            continue

                        dt = (self.tgt.time_index[1] - self.tgt.time_index[0]).total_seconds()
                        t_vals = np.arange(end - start) * dt
                        # H = 0
                        A = H / 2 * 0.25
                        L = 1.56*g*T**2/(2*np.pi)
                        omega = 2 * np.pi / T
                        wave_segment = A * np.cos(omega * t_vals)
                        wave_segment = wave_segment.reshape(len(wave_segment), 1)   

                        all_values = self.tgt.all_values(copy=True)
                        comp_idx = self.tgt.components.get_loc(comp)
                        all_values[start:end, comp_idx] = wave_segment + np.mean(all_values[start:end, comp_idx])
                        self.tgt = self.tgt.with_values(all_values)
                        value = {"H": H, "T": T}
                    else:
                        raise ValueError("Other mask types not implemented yet.")

                    self.maskLog.append({
                        "components": [comp],
                        "type": mask_type,
                        "value": value,
                        "temporal_range": [start, end]
                    })

    def maskAll_LSTM(self, historical_forecast=None, prbN=10, mask_type="substitute_LSTM"):
        if historical_forecast is None:
            raise ValueError("historical_forecast must be provided.")

        forecast_index = historical_forecast.time_index
        n_forecast = len(forecast_index)

        # Work on a copy of the mask log to avoid modifying while iterating
        original_mask_log = self.maskLog.copy()

        for mask in original_mask_log:
            comp = mask["components"][0]
            if comp not in self.tgt.components[-prbN:]:
                continue

            if mask["type"] == "factor":
                start, end = mask["temporal_range"]

                if end > n_forecast:
                    continue  # Skip if the forecast does not cover the range

                try:
                    forecasted_vals = historical_forecast[comp][start:end].values(copy=True)
                except:
                    continue  # Component missing or range error

                # Apply substitution in self.tgt2B, zero out self.tgt
                all_vals = self.tgt.all_values(copy=True)
                comp_idx = self.tgt.components.get_loc(comp)
                all_vals[start:end, comp_idx] = forecasted_vals

                self.tgt = self.tgt.with_values(all_vals)

                # Log this LSTM-based substitution
                self.maskLog.append({
                    "components": [comp],
                    "type": mask_type,
                    "value": "LSTM_prediction",
                    "temporal_range": [start, end],
                })



        

    def get_waveProperties(self, series, presentTime, history=100):
        dt = (series.time_index[1] - series.time_index[0]).total_seconds()
        n_steps = int(history / dt)
        
        start = max(presentTime - n_steps, 0)
        ts_window = series[start:presentTime]

        time = ts_window.time_index
        if len(time) > 0:
            time = np.array((time - time[0]).total_seconds())
        
        wse = ts_window.values(copy=True).flatten()
        T, H, _, _ = zero_up_crossing(time, wse)

        # Store average values in the arrays using component index
        return np.mean(T), np.mean(H)

@staticmethod
def maskFunction(
    data_array,
    cov_idx,
    tgt_idx,
    group_indices,
    group_to_idx,
):
    """
    Deterministically masks full groups over 5 equal-length segments of the time series.
    Segment 0: No mask
    Segment 1: Mask group A
    Segment 2: Mask group B
    Segment 3: Mask group C
    Segment 4: Mask group D
    """
    n_time = data_array.shape[0]
    n_groups = len(group_indices)
    segment_length = n_time // 5

    # Prepare output arrays
    masked_cov = data_array[:, cov_idx, :].copy()
    masked_tgt = data_array[:, tgt_idx, :].copy()
    switch_array = np.ones((n_time, n_groups, 1), dtype=np.float32)

    # Group masking across 5 segments
    for i, group_name in enumerate([None] + list(group_indices.keys())):
        start = i * segment_length
        end = (i + 1) * segment_length if i < 4 else n_time  # last segment may be longer

        if group_name is None:
            continue  # no masking in first segment

        # Mask both cov and tgt for channels in this group
        for ch in group_indices[group_name]:
            if ch in cov_idx:
                masked_cov[start:end, cov_idx.index(ch), :] = 0.0
            if ch in tgt_idx:
                masked_tgt[start:end, tgt_idx.index(ch), :] = 0.0

        # Set switch to 0.0 in this segment for the masked group
        switch_array[start:end, group_to_idx[group_name], :] = 0.0

    # Append switches to covariates
    masked_cov_with_switch = np.concatenate([masked_cov, switch_array], axis=1)

    return masked_cov_with_switch, masked_tgt

@staticmethod
def mask_ts(ts, components, ind=None, mask_factor=0.0):
    """
    Masks the specified components of a TimeSeries with a given value.
    
    Parameters:
    - ts: TimeSeries object to be masked.
    - components: List of component names to be masked.
    - mask_factor: Factor to use for masking (default is 0.0).

    Returns:
    - Masked TimeSeries object.
    """
    if isinstance(components, str):
        components = [components]
    
    values = ts.values(copy=True)  # shape: [time, component]
    comp_indices = [ts.components.get_loc(c) for c in components]
    if ind is None:
        ind[0] = 0
        ind[1] = -1
    values[ind[0]:ind[1], comp_indices] *= mask_factor

    return TimeSeries.from_times_and_values(ts.time_index, values, columns=ts.components)

