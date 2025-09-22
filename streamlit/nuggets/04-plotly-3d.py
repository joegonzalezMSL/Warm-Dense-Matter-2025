import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Generate sample T-ρ-P data
T = np.random.uniform(1, 100, 50)  # eV
rho = np.random.uniform(0.1, 10, 50)  # g/cm³
P = T * rho * 1.38e-16  # Simple pressure

fig = go.Figure(data=go.Scatter3d(x=T, y=rho, z=P, mode='markers'))
fig.update_layout(scene=dict(xaxis_title='Temperature (eV)', 
                            yaxis_title='Density (g/cm³)', 
                            zaxis_title='Pressure (Pa)'))
st.plotly_chart(fig)




