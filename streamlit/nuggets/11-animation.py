import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time

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
    