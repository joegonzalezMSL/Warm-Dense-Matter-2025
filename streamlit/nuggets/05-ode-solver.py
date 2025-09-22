import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def plasma_cooling(T, t, gamma=1.67):
    return -T**(gamma+1)  # Simplified cooling

t = np.linspace(0, 10, 100)
T0 = st.slider("Initial Temperature (eV)", 10, 1000, 100)
solution = odeint(plasma_cooling, T0, t)

fig, ax = plt.subplots()
ax.plot(t, solution)
ax.set_xlabel("Time (ps)")
ax.set_ylabel("Temperature (eV)")
st.pyplot(fig)
