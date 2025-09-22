import streamlit as st

options = ['Carbon', 'Boron', 'Iron',  'Silicon-Carbide']
 
st.title('Multiselect')

# Multi-select dropdown
selected_comps = st.multiselect(
    'Select one or more compositions',
    options,
    default=['Carbon']
)
if selected_comps:
    st.write(f'Selected compositions: {", ".join(selected_comps)}')
    

