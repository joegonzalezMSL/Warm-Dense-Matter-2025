import streamlit as st
import pandas as pd
import h5py
import numpy as np

file = st.file_uploader("Upload data", type=['csv', 'h5', 'txt'])
if file:
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
        st.dataframe(df)
    elif file.name.endswith('.h5'):
        # Handle HDF5 (common in simulations)
        st.write("HDF5 file detected - keys would be listed here")
        