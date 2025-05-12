import numpy as np

def zero_up_crossing(time, wse):
    """
    Identifies zero-upcrossings in a wave signal.
    
    Parameters:
    - time: array-like, time vector (1D)
    - wse: array-like, water surface elevation (1D)
    
    Returns:
    - T: array of periods between zero-upcrossings
    - H: array of wave heights (range between crossings)
    - stime: times at which zero-upcrossings occur
    - sfinder: indices of those zero-upcrossings
    """
    time = np.asarray(time).flatten()
    wse = np.asarray(wse).flatten()
    
    # Find zero-upcrossing indices
    crossings = np.where((wse[:-1] < 0) & (wse[1:] > 0))[0] + 1
    stime = time[crossings]
    sfinder = crossings

    # Calculate wave periods and heights
    T = np.diff(stime)
    H = [np.ptp(wse[sfinder[i]:sfinder[i+1]]) for i in range(len(sfinder)-1)]

    return T, np.array(H), stime, sfinder
