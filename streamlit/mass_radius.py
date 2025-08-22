import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 6.6743e-11  # m^3 kg^-1 s^-2
ME = 5.972e24   # kg
RE = 6.371e6    # m

# Compositions with parameters from Table 3
compositions = {
    'Carbon': {'rho0': 2250.0, 'c': 0.00350, 'n': 0.514},
    'Boron': {'rho0': 2340.0, 'c': 0.00350, 'n': 0.514},  # Approximated using Carbon's c and n, adjusted rho0
    'Iron': {'rho0': 8300.0, 'c': 0.00349, 'n': 0.528},
    'Silicon-Carbide': {'rho0': 3220.0, 'c': 0.00172, 'n': 0.537},
    'Water Ice': {'rho0': 1460.0, 'c': 0.00311, 'n': 0.513},
    'Gaseous (H/He)': None  # Special handling
}

st.title('Planet Mass-Radius Relation')

# Multi-select dropdown
selected_comps = st.multiselect(
    'Select one or more compositions',
    list(compositions.keys()),
    default=['Iron', 'Water Ice']
)

# Mass-Radius Plot
if selected_comps:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for comp in selected_comps:
        if comp == 'Gaseous (H/He)':
            # Empirical approximation: nearly constant radius for gas giants
            mass_me = np.logspace(2, 4, 100)  # 100 to 10000 M_Earth
            radius_re = 11.2 * (mass_me / 318.0)**0.01  # Slightly varying, normalized to Jupiter
            ax.plot(mass_me, radius_re, label=comp)
        else:
            params = compositions[comp]
            # Define rho function
            def rho_func(P):
                if P <= 0:
                    return 0.0
                return params['rho0'] + params['c'] * P**params['n']
            
            # Arrays for masses and radii
            log_pc_values = np.linspace(10, 25, 30)  # log10(Pc) in Pa
            masses = []
            radii = []
            
            for log_pc in log_pc_values:
                Pc = 10**log_pc
                # Define derivatives
                def fun(r, y):
                    m, P = y
                    rho = rho_func(P)
                    dmdr = 4 * np.pi * r**2 * rho
                    dPdr = -G * m * rho / r**2 if r > 1e-6 else 0.0
                    return [dmdr, dPdr]
                
                # Event to stop when P=0
                def event(r, y):
                    return y[1]
                event.terminal = True
                
                # Solve
                sol = solve_ivp(fun, [1e-6, 1e8], [0, Pc], method='RK45', rtol=1e-6, atol=1e-6, events=event)
                
                if sol.success and len(sol.t_events[0]) > 0:
                    R = sol.t[-1]
                    M = sol.y[0, -1]
                    if M > 0 and R > 0:
                        masses.append(M / ME)
                        radii.append(R / RE)
            
            if masses:
                masses = np.array(masses)
                radii = np.array(radii)
                # Sort by mass
                idx = np.argsort(masses)
                ax.plot(masses[idx], radii[idx], label=comp)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mass (Earth masses)')
    ax.set_ylabel('Radius (Earth radii)')
    ax.set_title('Mass-Radius Relations for Different Planetary Compositions')
    ax.legend()
    ax.grid(True, which='both', ls='--')
    
    st.pyplot(fig)

    # Pressure-Density Hugoniot Plot
    st.subheader('Pressure-Density Hugoniot')

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for comp in selected_comps:
        if comp != 'Gaseous (H/He)':
            params = compositions[comp]
            # Generate pressure values
            Ps = np.logspace(9, 15, 100)  # Pa, from 1 GPa to 1000 TPa
            rhos_kg = params['rho0'] + params['c'] * Ps ** params['n']
            rhos_gcm3 = rhos_kg / 1000.0
            Ps_TPa = Ps / 1e12
            mask = (rhos_gcm3 >= 1) & (rhos_gcm3 <= 14)
            ax2.plot(rhos_gcm3[mask], Ps_TPa[mask], label=comp)

    ax2.set_xlabel('Density (g/cm³)')
    ax2.set_ylabel('Pressure (TPa)')
    # ax2.set_yscale('log')
    ax2.set_title('Pressure-Density Relations for Selected Compositions')
    ax2.legend()
    ax2.grid(True, which='both', ls='--')

    st.pyplot(fig2)
else:
    st.write('Please select at least one composition.')
