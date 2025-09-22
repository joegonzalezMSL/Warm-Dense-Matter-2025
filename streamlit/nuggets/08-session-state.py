import streamlit as st

if 'results' not in st.session_state:
    st.session_state.results = []

# Get temperature input
temperature = st.number_input("Temperature", min_value=0.0, value=100.0)

if st.button("Run Calculation"):
    # Perform calculation only when button is clicked
    result = temperature ** 1.5
    st.session_state.results.append(result)

st.write("Calculation History:")
for i, r in enumerate(st.session_state.results):
    st.write(f"Run {i+1}: {r:.2f}")
    