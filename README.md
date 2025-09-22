# Warm-Dense-Matter-2025
This is the repository for the python codes, jupyter notebooks and other material related to the computational portion of the the course.


## LAMMPS                
Directory containing simple Molecular Dynamics Simulations. 

* NPT - Directory with a simple LAMMPS input deck to compress copper to 20 GPa and hold at 300 K.

* install_run_lammps_colab.ipynb - Short jupyter notebook on how to install and run LAMMPS in a Google Colab session.

## streamlit             
Directory containing simple apps demonstrating some of the many capabilities offered by Streamlit. This will serve as one of your reference for building apps to complete the homework during the course.

* mass_radius.py - Simple streamlit app to display static figures. In this case, display the Mass-Radius relation and Pressure-Density Hugoniot of specific elements
* pic-streamlit.py - App to demonstrate how to include simulation code into the streamlit app. In this case, it is a 1D plasma instability simulation.
* sph-streamlit.py - App to demonstrate how to include simulation code into the streamlit app. In this case, it is a Smoothed-Particle Hydrodynamics simulation of a collapsing star. 



# Project Directory Structure

The following is the directory structure for the `wdm-streamlit-nuggets` repository, showcasing an organized layout for Streamlit applications:

```bash
wdm-streamlit-nuggets/
├── nuggets/
│   ├── 01-file-reader.py
│   ├── 02-plotly-3d.py
│   ├── 03-file-saver.py
│   ├── 04-tabs-sidebar.py
│   ├── 05-ode-solver.py
│   ├── 06-parameter-sweep.py
│   ├── 07-multiformat-files.py
│   └── 08-session-state.py
├── homework/
└── combined-examples/
