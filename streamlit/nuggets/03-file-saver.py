import streamlit as st
import pandas as pd

data = {'Temperature': [1, 2, 3], 'Density': [0.5, 1.0, 2.0]}
df = pd.DataFrame(data)
st.dataframe(df)

csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, "results.csv", "text/csv")
