import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vmod.zero_crossing import zero_up_crossing as zuc
from scipy.integrate import simpson

plt.rcParams['font.family'] = 'Times New Roman'
def solve_sigma(t_target, t0=0, epsilon=0.01):
    """
    Solve for sigma given decay location and desired amplitude.
    
    Parameters:
    - x_target: location where amplitude ~ epsilon
    - x0: initial location of wave group (default 0)
    - epsilon: target amplitude fraction (e.g., 0.01 for 1%)

    Returns:
    - sigma: the Gaussian envelope standard deviation
    """
    return abs(t_target - t0) / np.sqrt(-2 * np.log(epsilon))

def gaussian_envelope(t, t0=0, sigma=400):
    """
    Compute Gaussian envelope transparency based on distance.
    
    Parameters:
    - x: array of positions
    - x0: center of envelope (default 0)
    - sigma: standard deviation of envelope (spread of the energy)
    
    Returns:
    - alpha: array of alpha values (0 to 1)
    """

    return np.exp(-((t - t0)**2) / (2 * sigma**2))

test = "group_velocity"
wave = "4"
if not os.path.exists(os.path.join("figures", f"{test}")):
    os.makedirs(os.path.join("figures", f"{test}"))

buoyFile = os.path.join("data", "FOCAL_wavedata", "scaledup", f"IR-{wave}.csv")
buoyData = pd.read_csv(buoyFile)

time = buoyData["Time"].values
wave1 = buoyData["wave1"].values
wave2 = buoyData["wave2"].values
wave3 = buoyData["wave3"].values
wave4 = buoyData["wave4"].values
wave5 = buoyData["wave5"].values    
# Constants
g = 9.81  # m/s^2

# For a given window of 20 seconds, calculate the zero-upcrossing, the maximum and minimum group velocity
windowT = 50  # seconds
window = int(windowT / (time[1] - time[0]))  # convert to number of samples
start_time = 0
cg_max1 = []
cg_max2 = []
cg_max3 = []
cg_max4 = []
cg_max5 = []
cg_min1 = []
cg_min2 = []
cg_min3 = []
cg_min4 = []
cg_min5 = []

for i in range(start_time, len(time), window):
    end_time = min(i + window, len(time))
    time_window = time[i:end_time]
    wave1_window = wave1[i:end_time]
    wave2_window = wave2[i:end_time]
    wave3_window = wave3[i:end_time]
    wave4_window = wave4[i:end_time]
    wave5_window = wave5[i:end_time]
    # Calculate zero-upcrossing
    T1, H1, stime1, sfinder1 = zuc(time_window, wave1_window)
    T2, H2, stime2, sfinder2 = zuc(time_window, wave2_window)
    T3, H3, stime3, sfinder3 = zuc(time_window, wave3_window)
    T4, H4, stime4, sfinder4 = zuc(time_window, wave4_window)
    T5, H5, stime5, sfinder5 = zuc(time_window, wave5_window)

    lambda_1 = g * T1**2 / (2 * np.pi)
    lambda_2 = g * T2**2 / (2 * np.pi)
    lambda_3 = g * T3**2 / (2 * np.pi)
    lambda_4 = g * T4**2 / (2 * np.pi)
    lambda_5 = g * T5**2 / (2 * np.pi)  
    k1 = 2 * np.pi / lambda_1
    k2 = 2 * np.pi / lambda_2
    k3 = 2 * np.pi / lambda_3
    k4 = 2 * np.pi / lambda_4
    k5 = 2 * np.pi / lambda_5
    c1 = np.sqrt(g / k1)  # phase velocity
    c2 = np.sqrt(g / k2)  # phase velocity
    c3 = np.sqrt(g / k3)  # phase velocity
    c4 = np.sqrt(g / k4)  # phase velocity
    c5 = np.sqrt(g / k5)  # phase velocity
    cg1 = c1 / 2  # group velocity
    cg2 = c2 / 2  # group velocity
    cg3 = c3 / 2  # group velocity
    cg4 = c4 / 2  # group velocity
    cg5 = c5 / 2  # group velocity
    if len(cg1) > 0:
        cg_max1.append(np.max(cg1))
        cg_min1.append(np.min(cg1))
    if len(cg2) > 0:
        cg_max2.append(np.max(cg2))
        cg_min2.append(np.min(cg2))
    if len(cg3) > 0:
        cg_max3.append(np.max(cg3))
        cg_min3.append(np.min(cg3))
    if len(cg4) > 0:
        cg_max4.append(np.max(cg4))
        cg_min4.append(np.min(cg4))
    if len(cg5) > 0:
        cg_max5.append(np.max(cg5))
        cg_min5.append(np.min(cg5))
cg_max1_mean = np.mean(cg_max1)
cg_max2_mean = np.mean(cg_max2)
cg_max3_mean = np.mean(cg_max3)
cg_max4_mean = np.mean(cg_max4)
cg_max5_mean = np.mean(cg_max5)
cg_min1_mean = np.mean(cg_min1)
cg_min2_mean = np.mean(cg_min2)
cg_min3_mean = np.mean(cg_min3)
cg_min4_mean = np.mean(cg_min4)
cg_min5_mean = np.mean(cg_min5)
cg_max_all = [cg_max1_mean, cg_max2_mean, cg_max3_mean, cg_max4_mean, cg_max5_mean]
cg_min_all = [cg_min1_mean, cg_min2_mean, cg_min3_mean, cg_min4_mean, cg_min5_mean]

cg_MAX = np.max(cg_max_all)
cg_MIN = np.min(cg_min_all)

lower_omega = g/(2*cg_MAX)
higher_omega = g/(2*cg_MIN)

Tp = 9.02  # for IR-1
omega_p = 2*np.pi/Tp

lower_omega_factor = lower_omega / omega_p
higher_omega_factor = higher_omega / omega_p

print(f"lower omega: {lower_omega} \n")
print(f"higher omega: {higher_omega} \n")
print(f"lower omega factor: {lower_omega_factor} \n")
print(f"higher omega factor: {higher_omega_factor} \n")

x1 = 0
x2 = x1 + 28.56
x3 = x1 + 144.83
x4 = x1 + 180.95
x5 = x1 + 302.47
xMax = 1000
x = np.linspace(0, xMax, 100)

starts = [x1, x2, x3, x4, x5]

cg_max_list = [cg_max1, cg_max2, cg_max3, cg_max4, cg_max5]
cg_min_list = [cg_min1, cg_min2, cg_min3, cg_min4, cg_min5]

windowT = 25
# plot the lines
probes = "P1, P3, P5"
i = 0
for cg_max, cg_min in zip(cg_max1, cg_min1):
    line_min = 1/cg_min * (x - starts[i]) - windowT
    line_max = 1/cg_max * (x - starts[i])
    intercept_point = (windowT) / (1/cg_min - 1/cg_max) + starts[i]
    plt.fill_between(x, line_min, line_max, where=(x <= intercept_point), color='gray', alpha=0.01)
# i = 1
# for cg_max, cg_min in zip(cg_max2, cg_min2):
#     line_min = 1/cg_min * (x - starts[i]) - windowT
#     line_max = 1/cg_max * (x - starts[i])
#     intercept_point = (windowT) / (1/cg_min - 1/cg_max) + starts[i]
#     plt.fill_between(x, line_min, line_max, where=(x <= intercept_point), color='gray', alpha=0.01)
i = 2
for cg_max, cg_min in zip(cg_max3, cg_min3):
    line_min = 1/cg_min * (x - starts[i]) - windowT
    line_max = 1/cg_max * (x - starts[i])
    intercept_point = (windowT) / (1/cg_min - 1/cg_max) + starts[i]
    plt.fill_between(x, line_min, line_max, where=(x <= intercept_point), color='gray', alpha=0.01)
# i = 3
# for cg_max, cg_min in zip(cg_max4, cg_min4):
#     line_min = 1/cg_min * (x - starts[i]) - windowT
#     line_max = 1/cg_max * (x - starts[i])
#     intercept_point = (windowT) / (1/cg_min - 1/cg_max) + starts[i]
#     plt.fill_between(x, line_min, line_max, where=(x <= intercept_point), color='gray', alpha=0.01)
i = 4
for cg_max, cg_min in zip(cg_max5, cg_min5):
    line_min = 1/cg_min * (x - starts[i]) - windowT
    line_max = 1/cg_max * (x - starts[i])
    intercept_point = (windowT) / (1/cg_min - 1/cg_max) + starts[i]
    plt.fill_between(x, line_min, line_max, where=(x <= intercept_point), color='gray', alpha=0.01)


plt.vlines(x5, ymin=0, ymax=100, color='black', label='Location of Interest', linestyles='--')
plt.xlabel('Space (m)')
plt.ylabel('Time (s)')
plt.xlim(0, 1000)
plt.ylim(0, 100)
plt.title(probes)
# plt.legend()
plt.grid()
plt.savefig(os.path.join("figures", f"{test}", f"IR{wave}_T_{windowT}_P_{probes}.pdf"), format="pdf")

plt.show()

# Calculate DPZ areas
probe_labels = [f'P{i+1}' for i in range(5)]

# Store areas for each probe
areas_per_probe = [[] for _ in range(5)]

# Compute areas
for i, (cg_max_vals, cg_min_vals) in enumerate(zip(cg_max_list, cg_min_list)):
    for cg_max, cg_min in zip(cg_max_vals, cg_min_vals):
        line_min = 1 / cg_min * (x - starts[i]) - windowT
        line_max = 1 / cg_max * (x - starts[i])
        intercept_point = (windowT) / (1 / cg_min - 1 / cg_max) + starts[i]
        mask = x <= intercept_point
        area = simpson(line_max[mask] - line_min[mask], x[mask])
        areas_per_probe[i].append(area)

# Plotting the area evolution per probe
plt.figure(figsize=(10, 6))
for i, areas in enumerate(areas_per_probe):
    plt.plot(range(len(areas)), areas, label=probe_labels[i])

plt.xlabel('Instance')
plt.ylabel(r'$A_{DPZ} (m.s)$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()