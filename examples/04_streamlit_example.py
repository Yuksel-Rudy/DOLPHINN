import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from vmod.dolphinn import DOLPHINN as DOL

# Configure
TEST = "streamlit example"
DOLPHINN_PATH = os.path.join("..", "saved_models", "1d", "IR4_model")
DATA_PATH = os.path.join("..", "data", "FOCAL_wavedata", "scaledup", "IR-4_testseed.csv")
FINAL_TIME = 500

# call dolphinn
dol = DOL()
dol.load(DOLPHINN_PATH)
data = pd.read_csv(DATA_PATH)
dofs = data[dol.dof].mul(dol.conversion, axis=1)

# Streamlit layout
st.title('Real-time Wave Prediction')

# Initialize plot in session state
if 'fig' not in st.session_state or 'ax' not in st.session_state:
    st.session_state.fig, st.session_state.ax = plt.subplots()
plot_container = st.empty()


def run_real_time():
    for time_user in range(0, FINAL_TIME + 1):
        time_user_idx = np.where(np.min(np.abs(data['Time'] - time_user)) == np.abs(data['Time'] - time_user))[0][0]
        t = data['Time'].iloc[0:time_user_idx].reset_index(drop=True)
        past_wave = dofs.iloc[0:time_user_idx]

        # Update the plot
        st.session_state.ax.clear()
        st.session_state.ax.plot(t, past_wave["wave5"], color='black', label='Wave')
        st.session_state.ax.set_xlabel('Time (s)')
        st.session_state.ax.set_ylabel('Wave Amplitude')
        st.session_state.ax.set_title('Real-time Wave Data')

        # Perform prediction
        ph = dol.time_horizon
        t2_idx = np.where(np.min(np.abs(data['Time'] - ph)) == np.abs(data['Time'] - ph))[0][0]
        if time_user_idx + t2_idx < len(data):
            try:
                t_pred, y_pred = dol.wrp_predict(data['Time'].iloc[0:time_user_idx + t2_idx],
                                                 dofs.iloc[0:time_user_idx], history=25)
                st.session_state.ax.scatter(t_pred, y_pred["wave5"], color='red', label='Wave Prediction')
                st.session_state.ax.legend()
            except Exception as e:
                st.error(f"Error during prediction: {e}")

        plot_container.pyplot(st.session_state.fig)
        st.session_state.ax.legend()
        time.sleep(0.1)


# Buttons for control
if st.button('Run Real-time'):
    run_real_time()
