import streamlit as st
import pandas as pd

st.title("Data Uploader")
uploaded_file = st.file_uploader("Choose CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} rows")
    st.dataframe(df.head())
    

