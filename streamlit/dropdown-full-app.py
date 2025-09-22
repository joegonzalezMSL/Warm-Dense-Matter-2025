import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import plotly.graph_objects as go
import io
import time

##Author: Saswat Mishra
 
st.set_page_config(page_title="Building blocks for your app", layout="centered")
 
# --- Menu ---
choice = st.sidebar.selectbox(
    "Choose a nugget",
    [
        "1: Write",
        "2: File Reader",
        "3: Multiselect",
        "4: 3D Rotatable Plot",
        "5: File Saver",
        "6: Tabs + Sidebar",
        "7: ODE Solver",
        "8: Real-time Parameter Sweep",
        "9: Multi-format File Handler",
        "10: Session State",
        "11: Animation",
    ]
)
 
### **Nugget 1: Write**
if choice.startswith("1:"):
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
 
### **Nugget 2: File Reader**
elif choice.startswith("2:"):
    st.title("Data Uploader")
 
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
 
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} rows")
        st.dataframe(df.head())
 
### **Nugget 3: Multiselect**
elif choice.startswith("3:"):
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
 
### **Nugget 4: 3D Rotatable Plot**
elif choice.startswith("4:"):
    st.title("3D Rotatable Plot")
 
    # Generate sample T-ρ-P data
    T = np.random.uniform(1, 100, 50)  # eV
    rho = np.random.uniform(0.1, 10, 50)  # g/cm³
    P = T * rho * 1.38e-16  # Simple pressure
 
    fig = go.Figure(data=go.Scatter3d(x=T, y=rho, z=P, mode='markers'))
    fig.update_layout(width=800,height=600,
        scene=dict( 
            xaxis_title='Temperature (eV)',
            yaxis_title='Density (g/cm³)',
            zaxis_title='Pressure (Pa)'
    ))
    st.plotly_chart(fig, use_container_width=True)
 
### **Nugget 5: File Saver**
elif choice.startswith("5:"):
    st.title("File Saver")
 
    data = {'Temperature': [1, 2, 3], 'Density': [0.5, 1.0, 2.0]}
    df = pd.DataFrame(data)
    st.dataframe(df)
 
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "results.csv", "text/csv")
 
### **Nugget 6: Tabs + Sidebar**
elif choice.startswith("6:"):
    st.title("Tabs + Sidebar")
 
    # Sidebar controls
    st.sidebar.header("Parameters")
    temp = st.sidebar.slider("Temperature (eV)", 1, 100, 10)
 
    # Main content with tabs
    tab1, tab2 = st.tabs(["EOS", "Transport"])
    with tab1:
        st.write(f"Equation of State at {temp} eV")
    with tab2:
        st.write(f"Transport properties at {temp} eV")
 
### **Nugget 7: ODE Solver**
elif choice.startswith("7:"):
    st.title("ODE Solver")
 
    def plasma_cooling(T, t, gamma=1.67):
        return -T**(gamma+1) # Simplified cooling
   
    t = np.linspace(0, 10, 100)
    T0 = st.slider("Initial Temperature (eV)", 10, 1000, 100)
    solution = odeint(plasma_cooling, T0, t)
    fig, ax = plt.subplots()
    ax.plot(t, solution)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Temperature (eV)")
    st.pyplot(fig)
 
### **Nugget 8: Real-time Parameter Sweep**
elif choice.startswith("8:"):
    st.title("Real-time Parameter Sweep")
 
    rho = st.slider("Density (g/cm³)", 0.1, 10.0, 1.0)
    temps = np.linspace(1, 100, 50)
    pressures = temps * rho * 1.38e-16  # Ideal gas law
 
    fig, ax = plt.subplots()
    ax.loglog(temps, pressures)
    ax.set_xlabel("Temperature (eV)")
    ax.set_ylabel("Pressure (Pa)")
    st.pyplot(fig)
 
### **Nugget 9: Multi-format File Handler**
elif choice.startswith("9:"):
    st.title("Multi-format File Handler")
    file = st.file_uploader("Upload data", type=['csv', 'h5', 'txt'])
    if file:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
            st.dataframe(df)
        elif file.name.endswith('.h5'):
            # Handle HDF5 (common in simulations)
            st.write("HDF5 file detected - keys would be listed here")
 
### **Nugget 10: Session State for Calculations**
 
elif choice.startswith("10:"):
    st.title("Session State for Calculations")
   
    T = st.number_input("Temperature")
 
    if 'results' not in st.session_state:
        st.session_state.results = []
 
    if st.button("Run Calculation"):
        result = T ** 1.5 # Some physics
        st.session_state.results.append(result)
 
    st.write("Calculation History:")
    for i, r in enumerate(st.session_state.results):
        st.write(f"Run {i+1}: {r:.2f}")
 

### **Nugget 11: Animation**
elif choice.startswith("11:"):
    st.title("Animation")
    st.write("""
    This example generates frames of a moving point along a sine curve and displays them as an animation.\n
    If you change the color of the point, the animation will regenerate with the new color only when you click `Run Simulation`.
    """)
    def run_simulation(c):
        frames = []
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
 
        # generate frames of a moving point
        for i in range(len(x)):
            fig, ax = plt.subplots()
            ax.plot(x, y, color="lightgray")             # static sine curve
            ax.scatter(x[i], y[i], color=c, s=80)        # moving point
            ax.set_xlim(0, 2*np.pi)
            ax.set_ylim(-1.2, 1.2)
 
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            frames.append(buf.read())
            plt.close(fig)
        return frames
   
    runSim = st.sidebar.button("Run Simulation")
    c = st.sidebar.color_picker("Pick a Color for the point", "#00f900")
    ## create a container for the animation
    image_placeholder = st.empty()
 
    if runSim:
        with st.spinner("Running simulation and generating animation..."):
            image_bytes = run_simulation(c)
            if image_bytes:
                st.write(f"Generated {len(image_bytes)} frames for animation")
                # Display animation
                for img in image_bytes:
                    image_placeholder.image(img, use_container_width=True)
                    time.sleep(0.01)  # Adjust speed of animation
            else:
                st.error("Simulation failed. Please check parameters and try again.")