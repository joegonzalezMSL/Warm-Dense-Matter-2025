import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import io
import time

def getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx):
    """
    Calculate the acceleration on each particle due to electric field
        pos      is an Nx1 matrix of particle positions
        Nx       is the number of mesh cells
        boxsize  is the domain [0,boxsize]
        n0       is the electron number density
        Gmtx     is an Nx x Nx matrix for calculating the gradient on the grid
        Lmtx     is an Nx x Nx matrix for calculating the laplacian on the grid
        a        is an Nx1 matrix of accelerations
    """
    try:
        N = pos.shape[0]
        dx = boxsize / Nx
        j = np.floor(pos / dx).astype(int)
        jp1 = j + 1
        weight_j = (jp1 * dx - pos) / dx
        weight_jp1 = (pos - j * dx) / dx
        jp1 = np.mod(jp1, Nx)  # periodic BC
        n = np.bincount(j[:, 0], weights=weight_j[:, 0], minlength=Nx)
        n += np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx)
        n *= n0 * boxsize / N / dx

        phi_grid = spsolve(Lmtx, n - n0, permc_spec="MMD_AT_PLUS_A")
        E_grid = -Gmtx @ phi_grid
        E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]
        a = -E
        return a
    except Exception as e:
        st.error(f"Error in getAcc: {str(e)}")
        return None

def run_simulation(N, dt, vb, tEnd, plot_freq):
    """Plasma PIC simulation with animation frames"""
    try:
        Nx = 400  # Number of mesh cells
        boxsize = 50  # periodic domain [0,boxsize]
        n0 = 1  # electron number density
        vth = 1  # beam width
        A = 0.1  # perturbation

        # Generate Initial Conditions
        np.random.seed(42)
        pos = np.random.rand(N, 1) * boxsize
        vel = vth * np.random.randn(N, 1) + vb
        Nh = int(N / 2)
        vel[Nh:] *= -1
        vel *= 1 + A * np.sin(2 * np.pi * pos / boxsize)

        # Construct matrix G (Gradient)
        dx = boxsize / Nx
        e = np.ones(Nx)
        diags = np.array([-1, 1])
        vals = np.vstack((-e, e))
        Gmtx = sp.spdiags(vals, diags, Nx, Nx)
        Gmtx = sp.lil_matrix(Gmtx)
        Gmtx[0, Nx - 1] = -1
        Gmtx[Nx - 1, 0] = 1
        Gmtx /= 2 * dx
        Gmtx = sp.csr_matrix(Gmtx)

        # Construct matrix L (Laplacian)
        diags = np.array([-1, 0, 1])
        vals = np.vstack((e, -2 * e, e))
        Lmtx = sp.spdiags(vals, diags, Nx, Nx)
        Lmtx = sp.lil_matrix(Lmtx)
        Lmtx[0, Nx - 1] = 1
        Lmtx[Nx - 1, 0] = 1
        Lmtx /= dx**2
        Lmtx = sp.csr_matrix(Lmtx)

        # Calculate initial accelerations
        acc = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx)
        if acc is None:
            return None

        # Number of timesteps
        Nt = int(np.ceil(tEnd / dt))
        t = 0

        # List to store image bytes for animation
        image_bytes = []

        # Simulation Main Loop
        for i in range(Nt):
            vel += acc * dt / 2.0
            pos += vel * dt
            pos = np.mod(pos, boxsize)
            acc = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx)
            if acc is None:
                return None
            vel += acc * dt / 2.0
            t += dt

            # Generate plot for every plot_freq timesteps
            if i % plot_freq == 0 or i == Nt - 1:
                fig, ax = plt.subplots(figsize=(5, 4), dpi=80)
                ax.scatter(pos[0:Nh], vel[0:Nh], s=0.4, color="blue", alpha=0.5)
                ax.scatter(pos[Nh:], vel[Nh:], s=0.4, color="red", alpha=0.5)
                ax.set_xlim(0, boxsize)
                ax.set_ylim(-6, 6)
                ax.set_xlabel("x")
                ax.set_ylabel("v")
                ax.set_title(f"Two-Stream Instability at t={t:.1f}")

                # Save plot to bytes
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=80)
                buf.seek(0)
                image_bytes.append(buf.getvalue())
                plt.close(fig)

        return image_bytes
    except Exception as e:
        st.error(f"Error in run_simulation: {str(e)}")
        return None

# Streamlit app
st.title("Plasma PIC Simulation: Two-Stream Instability with Animation")

# Sliders for parameters
dt = st.slider("Timestep (dt)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
vb = st.slider("Beam Velocity (vb)", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
N = st.slider("Number of Particles (N)", min_value=40000, max_value=100000, value=40000, step=1000)
tEnd = st.slider("End Time (tEnd)", min_value=50, max_value=1000, value=500, step=10)
plot_freq = st.slider("Plot Frequency (timesteps per frame)", min_value=1, max_value=100, value=10, step=1)

# Placeholder for animation
image_placeholder = st.empty()

# Run simulation when button is clicked
if st.button("Run Simulation"):
    with st.spinner("Running simulation and generating animation..."):
        image_bytes = run_simulation(N, dt, vb, tEnd, plot_freq)
        if image_bytes:
            st.write(f"Generated {len(image_bytes)} frames for animation")
            # Display animation
            for img in image_bytes:
                image_placeholder.image(img, use_container_width=True)
                time.sleep(0.1)  # Adjust speed of animation
        else:
            st.error("Simulation failed. Please check parameters and try again.")
            