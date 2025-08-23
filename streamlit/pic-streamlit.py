import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import io
import time

# Create Your Own Plasma PIC Simulation (With Python)
# Philip Mocz (2020) Princeton University, @PMocz

# Simulate the 1D Two-Stream Instability
# Code calculates the motions of electron under the Poisson-Maxwell equation
# using the Particle-In-Cell (PIC) method

## simulations from Philip Mocz
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

def run_simulation(N, dt, vb, tEnd, plot_freq,c1,c2):
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
                ax.scatter(pos[0:Nh], vel[0:Nh], s=0.5, color=c1, alpha=0.5)
                ax.scatter(pos[Nh:], vel[Nh:], s=0.5, color=c2, alpha=0.5)
                ax.set_xlim(0, boxsize)
                ax.set_ylim(-6, 6)
                ax.set_xlabel("Position")
                ax.set_ylabel("Velocity")
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


### Begin the streamlit part
## The order of commands is directly related to placement in the app

# Title displayed on webapp first
# st.markdown('### This is a simple web app built to demonstrate a 1D Plasma instability simulation')
st.title('''This is a simple web app built to demonstrate a 1D Plasma instability simulation''')

## Write some descriptive text about what your web app does using st.write() or st.markdown()
st.markdown('### 1D Plasma Instability Simulation - Philip Mocz @PMocz - Princeton University')
## multi line description, verbatim
st.write("""
    Simulate the 1D Instability of two electron beams traveling opposite to eachother.
    Code calculates the motions of electron under the Poisson-Maxwell equation
    using the Particle-In-Cell (PIC) method

""")

## streamlit can accept Latex formated text to display inline equations as well as separate equations
latext = r'''
Consider a one-dimensional system of electrons in an
unmagnetized uniform medium of ions. The electrons will be described by
${N}$ particles, indexed by ${i}$, each having a position ${r_i}$ and velocity ${v_i}$

The electrons feel an acceleration $a_{i}$ which is due to the electric field E at
the location of the particle. The equations of motion for the electrons are
given by:
$$ 
\frac{d r_{i}}{dt}=v_{i}
$$
$$
\frac{d v_{i}}{dt}=a_{i} = -E(r_{i})
$$ 

What remains to be done is to calculate the electric field ${E}$, which depends
on the number density ${n}$ of the electrons. The electric field ${E}$ is defined as
the negative gradient of the electric potential $\phi$, which is given by
Poisson's equation sourced by the density:

$$
E(x) = -\frac{d \phi(x)}{dx}
$$

$$
\frac{d^2 \phi(x)}{dx^2}=n-n{_0}
$$

where $n{_0}$ is the average electron density (a constant).
The Particle-In-Cell (PIC) Method computes the density and
subsequently the potential and electric field on a mesh. Hence it is called
a particle-mesh method. The electric field is then extrapolated onto the
location of the particles in order to advance them. 

'''
st.write(latext)

## Streamlit allows the embeding of hyperlinks, use these to provide citations or for guiding 
## the reader to learn more information
url = "https://medium.com/@philip-mocz/create-your-own-plasma-pic-simulation-with-python-39145c66578b"
st.write("See more detailed information on the PIC code by Philip on [Medium](%s)" %url)


## Sliders and text boxes to accept parameters to control the simulation
## they are stored in the sidebar to keep content in focus. They can also 
## be put in the main display by removing ".sidebar." from each of the below
st.sidebar.title("Define simulation parameters")
dt = st.sidebar.slider("Timestep (dt)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
vb = st.sidebar.slider("Beam Velocity (vb)", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
N  = st.sidebar.number_input(
    "Number of particles", value=10000,step=10000, placeholder="Type a number..."
)
tEnd  = st.sidebar.number_input(
    "Total simulation time to run", value=500,step=10, placeholder="Type a number..."
)
plot_freq = st.sidebar.slider("Plot Frequency (timesteps per frame)", min_value=1, max_value=100, value=10, step=1)

c1 = st.sidebar.color_picker("Pick A Color for Stream 1", "#00f900")
c2 = st.sidebar.color_picker("Pick A Color for Stream 2", "#8300F9")

runSim = st.sidebar.button("Run Simulation")

## create a container for the animation
image_placeholder = st.empty()

## Run simulation when button is clicked
if runSim:
    with st.spinner("Running simulation and generating animation..."):
        image_bytes = run_simulation(N, dt, vb, tEnd, plot_freq,c1,c2)
        if image_bytes:
            st.write(f"Generated {len(image_bytes)} frames for animation")
            # Display animation
            for img in image_bytes:
                image_placeholder.image(img, use_container_width=True)
                time.sleep(0.1)  # Adjust speed of animation
        else:
            st.error("Simulation failed. Please check parameters and try again.")
            