import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

rho = st.slider("Density (g/cm³)", 0.1, 10.0, 1.0)
temps = np.linspace(1, 100, 50)
pressures = temps * rho * 1.38e-16  # Ideal gas law

fig, ax = plt.subplots()
ax.loglog(temps, pressures)
ax.set_xlabel("Temperature (eV)")
ax.set_ylabel("Pressure (Pa)")
st.pyplot(fig)
