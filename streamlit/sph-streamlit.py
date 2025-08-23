import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import io
import time

# SPH Simulation Functions (from provided code)
def W(x, y, z, h):
    r = np.sqrt(x**2 + y**2 + z**2)
    w = (1.0 / (h * np.sqrt(np.pi))) ** 3 * np.exp(-(r**2) / h**2)
    return w

def gradW(x, y, z, h):
    r = np.sqrt(x**2 + y**2 + z**2)
    n = -2 * np.exp(-(r**2) / h**2) / h**5 / (np.pi) ** (3 / 2)
    wx = n * x
    wy = n * y
    wz = n * z
    return wx, wy, wz

def getPairwiseSeparations(ri, rj):
    M = ri.shape[0]
    N = rj.shape[0]
    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))
    riz = ri[:, 2].reshape((M, 1))
    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))
    rjz = rj[:, 2].reshape((N, 1))
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T
    return dx, dy, dz

def getDensity(r, pos, m, h):
    M = r.shape[0]
    dx, dy, dz = getPairwiseSeparations(r, pos)
    rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))
    return rho

def getPressure(rho, k, n):
    P = k * rho ** (1 + 1 / n)
    return P

def getAcc(pos, vel, m, h, k, n, lmbda, nu):
    N = pos.shape[0]
    rho = getDensity(pos, pos, m, h)
    P = getPressure(rho, k, n)
    dx, dy, dz = getPairwiseSeparations(pos, pos)
    dWx, dWy, dWz = gradW(dx, dy, dz, h)
    ax = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWx, 1).reshape((N, 1))
    ay = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWy, 1).reshape((N, 1))
    az = -np.sum(m * (P / rho**2 + P.T / rho.T**2) * dWz, 1).reshape((N, 1))
    a = np.hstack((ax, ay, az))
    a -= lmbda * pos
    a -= nu * vel
    return a

def run_simulation(N, tEnd, dt, M, R, h, k, nu, plot_every):
    try:
        t = 0
        n = 1  # polytropic index (fixed as in original code)
        np.random.seed(42)
        lmbda = (
            2
            * k
            * (1 + n)
            * np.pi ** (-3 / (2 * n))
            * (M * gamma(5 / 2 + n) / R**3 / gamma(1 + n)) ** (1 / n)
            / R**2
        )
        m = M / N
        pos = np.random.randn(N, 3)
        vel = np.zeros(pos.shape)
        acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)
        Nt = int(np.ceil(tEnd / dt))
        
        rr = np.zeros((100, 3))
        rlin = np.linspace(0, 1, 100)
        rr[:, 0] = rlin
        rho_analytic = lmbda / (4 * k) * (R**2 - rlin**2)
        
        pos_frames = []
        rho_part_frames = []
        rho_rad_frames = []
        times = []
        particle_images = []
        density_images = []
        
        def save_frame():
            pos_frames.append(pos.copy())
            rho_part_frames.append(getDensity(pos, pos, m, h))
            rho_rad_frames.append(getDensity(rr, pos, m, h))
            times.append(t)
        
        save_frame()  # initial frame
        
        for i in range(Nt):
            vel += acc * dt / 2
            pos += vel * dt
            acc = getAcc(pos, vel, m, h, k, n, lmbda, nu)
            vel += acc * dt / 2
            t += dt
            if (i + 1) % plot_every == 0 or i == Nt - 1:
                save_frame()
                
                # Generate particle plot
                fig_p = plt.figure(figsize=(4, 4), dpi=400)
                ax_p = fig_p.add_subplot(111)
                rho_i = rho_part_frames[-1]
                cval = np.minimum((rho_i - 3) / 3, 1).flatten()
                ax_p.scatter(pos[:, 0], pos[:, 1], c=cval, cmap=plt.cm.plasma, s=10, alpha=0.8)
                ax_p.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
                ax_p.set_aspect('equal', 'box')
                ax_p.set_xticks([-1, 0, 1])
                ax_p.set_yticks([-1, 0, 1])
                ax_p.set_facecolor((0.1, 0.1, 0.1))
                ax_p.set_title(f"Time: {t:.2f}")
                buf_p = io.BytesIO()
                fig_p.savefig(buf_p, format="png", dpi=400)
                buf_p.seek(0)
                particle_images.append(buf_p.getvalue())
                plt.close(fig_p)
                
                # Generate density plot
                fig_d = plt.figure(figsize=(4, 2), dpi=400)
                ax_d = fig_d.add_subplot(111)
                ax_d.plot(rlin, rho_analytic, color="gray", linewidth=2, label="Analytic")
                ax_d.plot(rlin, rho_rad_frames[-1], color="blue", label="Simulated")
                ax_d.set(xlim=(0, 2), ylim=(0, 10))
                ax_d.set_aspect(0.1)
                ax_d.set_xlabel("Radius")
                ax_d.set_ylabel("Density")
                ax_d.legend()
                ax_d.set_title(f"Time: {t:.2f}")
                buf_d = io.BytesIO()
                fig_d.savefig(buf_d, format="png", dpi=400)
                buf_d.seek(0)
                density_images.append(buf_d.getvalue())
                plt.close(fig_d)
        
        return particle_images, density_images
    except Exception as e:
        st.error(f"Error in run_simulation: {str(e)}")
        return None, None


# Title displayed on webapp first
# st.markdown('### This is a simple web app built to demonstrate a 1D Plasma instability simulation')
st.title('''This is a simple web app built to simulate a star using Smmothed Particle Hydrodynamics''')

## Write some descriptive text about what your web app does using st.write() or st.markdown()
st.markdown('### Create Your Own Smoothed-Particle Hydrodynamics Simulation - Philip Mocz @PMocz - Princeton University')
## multi line description, verbatim
st.write("""
    We will start with some initial condition and relax it into a stable stellar structure and measure the star's density 
    as a function of radius. The SPH method has many applications in astrophysics and elsewhere, and is in fact a general 
    one to simulate all types of fluids.

""")
st.markdown('### SPH Method')
st.markdown(r"""
    We will represent the fluid as a collection of N point particles, indexed by i=1,…,N. Each particle has:
    * Mass ${m}$
    * Position $r_{i}$ = [$x_{i},y_{i},z_{i}$]
    * Velocity $v_{i}$ = [$vx_{i},vy_{i},vz_{i}$]
    """)
st.write("The particles will experience motion according to the Euler equation for an idea fluid:")

latext = r'''
$$
\frac{d \textbf{v}}{dt}=-\frac{1}{\rho}\nabla P+\frac{1}{\rho}\textbf{f}
$$
Here, ${P}$ is the fluid pressure and $\textbf{f}$ is any additional external forces we may choose to impose on the system. 
We will assume that the pressure is given by a simple equation of state, called the polytropic equation of state (meaning it is just a function of density):
$$
P = k\rho^{1+1/n} 
$$
Here ${k}$ is a normalization constant and ${n}$ is the polytropic index.

For our simulation, we will consider the following external forces:
$$
\textbf{f} = -\lambda\textbf{r}-\nu\textbf{v}
$$

'''
st.write(latext)
st.write('''
    The first term is meant to represent the static gravitational potential of the star. 
    The second term represents viscosity, which adds friction that slows down particles. 
    We add this term so that the simulation can evolved to a steady state solution.
    ''')

url = "https://ui.adsabs.harvard.edu/abs/2004MNRAS.350.1449M/abstract"
st.write("The model in our tutorial is motivated from a simplified simulation of stars in the paper by [Monaghan & Price](%s)" %url)
url = "https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1"
st.write("See more detailed information on the SPH code by Philip on [Medium](%s)" %url)


# Sidebar for input parameters
with st.sidebar:
    st.title("Define simulation parameters")
    N = st.number_input("Number of Particles (N)", min_value=100, max_value=1000, value=400, step=50)
    tEnd = st.number_input("Simulation End Time (tEnd)", min_value=1.0, max_value=20.0, value=12.0, step=0.5)
    dt = st.number_input("Timestep (dt)", min_value=0.01, max_value=0.1, value=0.04, step=0.01)
    M = st.number_input("Star Mass (M)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    R = st.number_input("Star Radius (R)", min_value=0.1, max_value=2.0, value=0.75, step=0.05)
    h = st.number_input("Smoothing Length (h)", min_value=0.05, max_value=0.5, value=0.1, step=0.01)
    k = st.number_input("Equation of State Constant (k)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    nu = st.number_input("Damping (nu)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    plot_every = st.number_input("Plot Every N Timesteps", min_value=1, max_value=50, value=5, step=1, format="%d")
    speed = st.number_input("Speed of animation", min_value=0.0, max_value=1.0, value=0.2, step=0.05, format="%f")
    runSim = st.button("Run Simulation")

# Run simulation button
if runSim:
    with st.spinner("Running simulation and generating images..."):
        particle_images, density_images = run_simulation(N, tEnd, dt, M, R, h, k, nu, int(plot_every))
        if particle_images and density_images:
            st.write(f"Generated {len(particle_images)} frames for animation")
            particle_placeholder = st.empty()
            density_placeholder = st.empty()
            for p_img, d_img in zip(particle_images, density_images):
                particle_placeholder.image(p_img, caption="Particle Positions", use_container_width=True)
                density_placeholder.image(d_img, caption="Density Profile", use_container_width=True)
                time.sleep(speed)  # Adjust speed of animation
        else:
            st.error("Simulation failed. Please check parameters and try again.")