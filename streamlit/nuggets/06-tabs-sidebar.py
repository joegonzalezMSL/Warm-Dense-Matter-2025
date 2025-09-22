import streamlit as st

# Sidebar controls
st.sidebar.header("Parameters")
temp = st.sidebar.slider("Temperature (eV)", 1, 100, 10)

# Main content with tabs
tab1, tab2 = st.tabs(["EOS", "Transport"])
with tab1:
    st.write(f"Equation of State at {temp} eV")
with tab2:
    st.write(f"Transport properties at {temp} eV")
    