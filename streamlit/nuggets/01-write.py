import streamlit as st


st.title("Stylistic text formatting")
st.markdown('### HEADING')
    ## multi line description, verbatim
st.write("""
    Multi-line description, verbatim\n
    You can use st.write() or st.markdown() for text output.
   
    You can include **bold** or *italic* text, lists, and more.
         
    You can also include code snippets like `x = 5` or larger blocks:
    ```
    def example():
        return "Hello, Streamlit!"
    ```
""")

## LaTeX
st.markdown('### LaTeX')
latext = r'''
$$
\frac{d v_{i}}{dt}=a_{i} = -E(r_{i})
$$ 

'''
st.write(latext)

st.markdown('### URL')
url = "https://doc-plotly-chart.streamlit.app/"
st.write("See more [here](%s)" %url)
