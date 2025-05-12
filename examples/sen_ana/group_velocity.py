import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vmod.zero_crossing import zero_up_crossing as zuc

test = "group_velocity"
wave = "1"
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
window = 50  # seconds
window = int(window / (time[1] - time[0]))  # convert to number of samples
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
    print(f" {time_window[0]} s")
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




x1 = 0
x2 = x1 + 28.56
x3 = x1 + 144.83
x4 = x1 + 180.95
x5 = x1 + 302.47
tMax = 100
t = np.array([0, tMax])
l1_max = cg_max1_mean*t + x1
l2_max = cg_max2_mean*t + x2
l3_max = cg_max3_mean*t + x3
l4_max = cg_max4_mean*t + x4
l5_max = cg_max5_mean*t + x5

l1_min = cg_min1_mean*t + x1
l2_min = cg_min2_mean*t + x2
l3_min = cg_min3_mean*t + x3
l4_min = cg_min4_mean*t + x4
l5_min = cg_min5_mean*t + x5

plt.figure(figsize=(5, 5))
plt.plot(t, l1_max, color='blue', label='Max Group Velocity')
plt.plot(t, l2_max, color='blue')
plt.plot(t, l3_max, color='blue')
plt.plot(t, l4_max, color='blue')
plt.plot(t, l5_max, color='blue')
plt.plot(t, l1_min, color='red', label='Min Group Velocity', linestyle='--')
plt.plot(t, l2_min, color='red', linestyle='--')
plt.plot(t, l3_min, color='red', linestyle='--')
plt.plot(t, l4_min, color='red', linestyle='--')
plt.plot(t, l5_min, color='red', linestyle='--')

plt.hlines(x5, xmin=0, xmax=tMax, color='black', label='Location of Interest', linestyles='--')
plt.xlabel('Time (s)')
plt.ylabel('Location (m)')
plt.legend()
plt.grid()
plt.savefig(os.path.join("figures", f"{test}", f"IR-{wave}_group_velocity.pdf"), format="pdf")

