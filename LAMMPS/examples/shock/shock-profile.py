#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import re
import argparse
import textwrap


# Global Constants
mass = 63.5460  # Carbon atom mass in atomic mass units (amu)
amu_to_kg = 1.66053906660e-27  # 1 amu in kg
J_to_eV = 1 / 1.6e-19  # Conversion J to eV
angstrom_to_m = 1e-10  # Ångstroms to meters conversion
bar_to_gpa = 0.0001  # Conversion from bar to GPa
mass_carbon_kg = mass * amu_to_kg  # Atomic mass of carbon in kg
kb_ev = 8.617e-5  # Boltzmann constant in eV/K
kg_m3_to_g_cm3 = 0.001  # Conversion from kg/m^3 to g/cm^3
pressfactor = 160.219   # Pressure factor eV/A^3 ==> GPa
#/Constants

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_box_info(fName):
    if not os.path.exists(fName):
        eprint(f"{fName} not found.")
        return 0, 0
    with open(fName, 'r') as file:
        for line in file:
            if "box =" in line:
                line = line.strip()
                if "orthogonal box" in line:
                    box_values = line.split("box = ")[1].split("to")
                    x_min, y_min, z_min = map(float, box_values[0].strip("() ").split())
                    x_max, y_max, z_max = map(float, box_values[1].strip("() ").split())
                    z_length = z_max - z_min
                    xy_area = (x_max - x_min) * (y_max - y_min)
                    eprint(f"XY_Area {xy_area} Z_Length {z_length} A")
                    
            #region          box block 0 3 0 4 0 50
            if "region          box block" in line:
                box_values = line.split()
                nx = int(box_values[-5])
                ny = int(box_values[-3])
                nz = int(box_values[-1])
                eprint(f"Unit cell replications: {nx} {ny} {nz}")

    return z_min, z_length, xy_area, nx, ny, nz


def read_profile(fName, bin_volume_A3, min_step_to_read= -1E9, max_step_to_read=1E9):
    # dir1=str(dir1)
    # eprint(f"Processing profile file: [{dir1}] @ {vel1} km/s")  # Now actually reading 1D LAMMPS profiles
    # if not os.path.exists(fName):
    #     eprint(f"{fName} not found in folder {dir1}. Skipping.")
    #     return []
 # define bin_size    
    data_list = []  # Create a local list to collect data
    with open(fName, 'r') as file:
        lines = file.readlines()
    line_count=0
    current_timestep = None
    for line in lines:
        tokens = line.strip().split()
        if line.startswith("#") or not line or (tokens[0].isdigit() and len(tokens) == 3):
            continue
        line_count+=1 
        tokens = line.split()
        values = tokens
        data_list.append(float(values[1]))
        if line_count==2:
            bin_size_z= data_list[1]-data_list[0]  
            eprint(f" use automatic found bin-size: {bin_size_z} A")
            break     
    # bin_volume_A3 = xy_area * bin_size_z
    bin_volume_m3 = bin_volume_A3 * angstrom_to_m ** 3

    data_list = []  # Create a local list to collect data
    current_timestep = None
    line_count=0
    timeframes_count=0
    for line in lines:
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        tokens = line.split()
        if tokens[0].isdigit() and len(tokens) == 3:
            current_timestep = int(tokens[0])
            timeframes_count+=1
        elif current_timestep!=None:
            if current_timestep>max_step_to_read:
                eprint(f"skipping timesteps above: {max_step_to_read}")
                break
            values = tokens
            #    0     1      2      3        4         5           6            7      8  9  10   11   12    13
            # Chunk Coord1 Ncount c_KE_ALL c_PE_ALL c_SE_ALL[1] c_SE_ALL[2] c_SE_ALL[3] vx vy vz v_vx2 v_vy2 v_vz2
            z_coord = round(float(values[1]), 2)
            ncount = int(float(values[2]))
            ke = round(float(values[3]), 3)
            pe = round(float(values[4]), 3)
            stress_xx = float(values[5])
            stress_yy = float(values[6])
            stress_zz = float(values[7])
            v_com = round(float(values[10]) * 100, 2)  # Velocity of center of mass, m/s

            # Calculate density
            density = (ncount * mass_carbon_kg) / bin_volume_A3 /angstrom_to_m**3 * kg_m3_to_g_cm3

            # Calculate pressure and shear stresses
            p_xx = -round(stress_xx * ncount / bin_volume_A3, 2)
            p_yy = -round(stress_yy * ncount / bin_volume_A3, 2)
            p_zz = -round(stress_zz * ncount / bin_volume_A3, 2)
            pressure = -round((stress_xx + stress_yy + stress_zz) / 3 * ncount / bin_volume_A3, 2)

            # Correct temperature by subtracting COM velocity contribution
            ke_thermal = ke - 0.5 * (v_com ** 2) * mass_carbon_kg * J_to_eV
            temperature = 2 * ke_thermal / (3 * kb_ev)

            # Append the data to the list as a dictionary
            line_count+=1 # counting new data points to be addes 
            data_list.append({
                "timestep": int(current_timestep),
                "z-coordinate": z_coord,
                "zl": bin_size_z,
                "pressure": pressure * bar_to_gpa,
                "P_xx": p_xx * bar_to_gpa,
                "P_yy": p_yy * bar_to_gpa,               
                "P_zz": p_zz * bar_to_gpa,
                "ke": ke,
                "pe": pe,
                "e_tot": ke_thermal+pe,
                "v_com": v_com,
                "temperature": temperature,
                "density": density,
                "n_atoms": ncount
            })
    eprint (f" Datapoints added: {line_count}; timesteps added: {timeframes_count}; av bins/timesteps: {round(line_count/timeframes_count,1)}")
    #print (f"Min frame step: {data_list[0]['timestep']} ; Max frame step:  {data_list[-1]['timestep']}")
    eprint ("--------------------------")
    return data_list

def variable_rebinning(data_list,xy_area, N_atoms_target):
    eprint("Performing rebinning with fixed particle count in each bin")
    timesteps = []
    current_timestep_data = []
    current_timestep = data_list[0]["timestep"]
    # direction = data_list[0]["direction"]
    # velocity = data_list[0]["velocity"]
    layer_size = data_list[1]["z-coordinate"] - data_list[0]["z-coordinate"]

    # Calculate properties - averaging
    properties_N = ["z-coordinate", "ke", "pe", "e_tot", "v_com", "temperature"]
    # properties_V = ["density","pressure", "P_xx", "P_yy", "P_zz", "tau_xz", "tau_yz"]
    properties_V = ["density","pressure", "P_xx", "P_yy", "P_zz"]

    for entry in data_list:
        if entry["timestep"] != current_timestep:
            timesteps.append(current_timestep_data)
            current_timestep_data = []
            current_timestep = entry["timestep"]
        current_timestep_data.append(entry)

    # Add the last timestep's data
    if current_timestep_data:
        timesteps.append(current_timestep_data)

    data_rebinned = []
    line_count = 0
    timeframes_count = len(timesteps)

    for layers in timesteps:
        timestep = layers[0]["timestep"]  # Extract the timestep
        layers = sorted(layers, key=lambda x: x["z-coordinate"])
        current_bin = []
        current_atom_count = 0
        bin_size = 0

        for layer in layers:
            num_atoms = layer["n_atoms"]

            if current_atom_count + num_atoms <= N_atoms_target: #full bins
                # Fully add this layer to the current bin
                current_bin.append(layer)
                current_atom_count += num_atoms
                bin_size += layer_size
            else: #partial
                # Prepare to create a new bin
                line_count += 1
                excess_atoms = current_atom_count + num_atoms - N_atoms_target
                included_atoms = num_atoms - excess_atoms

                # Create partial layers
                partial_layer_current = {key: layer[key] for key in layer}
                partial_layer_next = {key: layer[key] for key in layer}

                # Update only n_atoms and zl for the partial layers
                partial_layer_current["n_atoms"] = included_atoms
                partial_layer_next["n_atoms"]    = excess_atoms
                partial_layer_current["zl"] = included_atoms / num_atoms * layer_size
                partial_layer_next["zl"] = excess_atoms / num_atoms * layer_size

                # Add the partial layer for the current bin
                current_bin.append(partial_layer_current)
                current_atom_count += included_atoms
                bin_size += partial_layer_current["zl"]
                if bin_size<2*layer_size:
                    eprint("Warning Too small number of atoms in bin, increase N_target! Algo works unstable!!") 

                # Perform weighted averaging for the current bin
                rebinned_entry = {
                    "timestep": timestep,
                    "n_atoms": current_atom_count,
                    "zl": bin_size
                }


                for prop in properties_N:
                    rebinned_entry[prop] = sum( layer[prop] * layer["n_atoms"] for layer in current_bin) / current_atom_count

                for prop in properties_V:
                    rebinned_entry[prop] = sum( layer[prop] * layer["zl"] for layer in current_bin ) / rebinned_entry["zl"]
              #  rebinned_entry["density"] = sum( layer["n_atoms"] for layer in current_bin)
                rebinned_entry["density"] = rebinned_entry["n_atoms"] * mass_carbon_kg/ (xy_area*rebinned_entry["zl"]) /angstrom_to_m**3 * kg_m3_to_g_cm3

                # Append the merged bin
                data_rebinned.append(rebinned_entry)

                # Start the next bin with the partial layer
                current_bin = [partial_layer_next] if excess_atoms > 0 else []
                current_atom_count = excess_atoms
                bin_size = partial_layer_next["zl"] if excess_atoms > 0 else 0

        if current_bin:
            current_bin.append(layer)
            current_atom_count += num_atoms
            bin_size += layer_size
            line_count += 1
            rebinned_entry = {
                "timestep": timestep,
                "n_atoms": current_atom_count,
                "zl": sum(layer["zl"] for layer in current_bin),
            }

            # Calculate density
            rebinned_entry["density"] = sum(layer["density"] * layer["zl"] for layer in current_bin ) / rebinned_entry["zl"]
            # Calculate properties
            for prop in properties_N:
                rebinned_entry[prop] = sum( layer[prop] * layer["n_atoms"] for layer in current_bin ) / current_atom_count
            for prop in properties_V:
                rebinned_entry[prop] = sum(   layer[prop] * layer["zl"] for layer in current_bin  ) / rebinned_entry["zl"]

            data_rebinned.append(rebinned_entry)

    # print(f"Datapoints reduced to: {line_count}; timesteps: {timeframes_count}; av. bins/timestep: {round(line_count / timeframes_count, 1)}")
    # print(f"Memory reduction: x {round(len(data_list) / line_count, 1)}")
    # print("--------------------------")
    eprint(f"Datapoints reduced to: {line_count}; timesteps: {timeframes_count}; av. bins/timestep: {round(line_count / timeframes_count, 1)}")
    eprint(f"Memory reduction: x {round(len(data_list) / line_count, 1)}")
    eprint("--------------------------")
    return data_rebinned



def get_arguments(argv):

    cli = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
           Utility to convert 2theta XRD patterns to q-space XRD
         -----------------------------------------------------------------
         '''),
    epilog=textwrap.dedent('''\
         examples:
         -----------------------------------------------------------------
            %(prog)s graphite.xrd > graphite-q2t.xrd
            %(prog)s --lammps graphite.xrd > graphite-q2t.xrd
            %(prog)s --versta graphite.xrd > graphite-q2t.xrd

         '''))

    cli.add_argument(
        "FILE",
        type=str,
        help="Specify the name of the profile")
    cli.add_argument(
        "-d","--direction",
        dest="dir",
        type=str,
        default="100",
        help="Specify the shock direction",
        action="store")
    cli.add_argument(
        "-l","--log-file",
        dest="logFile",
        default="log.lammps",
        help="Specify the lammps log file for reading the box dimensions",
        action="store")
    cli.add_argument(
        "-r","--replications",
        dest="reps",
        nargs=3,
        type=int,
        help="Specify each of the number of unit cells in each dimension i.e. -> -r 1 1 10",
        action="store")
    cli.add_argument(
        "-n","--nlocal",
        dest="cellAtoms",
        type=int,
        default=16,
        help="Specify number of atoms in a single unit cell",
        action="store")
    cli.add_argument(
        "-v","--velocity",
        dest="velocity",
        type=float,
        help="Specify the piston velocity in km/s",
        action="store")
    cli.add_argument(
        "-t","--time-step",
        dest="TS",
        type=int,
        default=40000,
        help="Specify the timestep to be plotted",
        action="store")
    cli.add_argument(
        '--version',
        action='version',
        version='%(prog)s 4.0.')
    args = cli.parse_args()

    return args

def main(argv):

    parameters = get_arguments( argv )

    # profileName = 'profiles.dat'  # Files to parse
    profileName = parameters.FILE
    logFile = parameters.logFile
    # nx = parameters.reps[0]
    # ny = parameters.reps[1]
    # nz = parameters.reps[2]
    n = 16
    TS = parameters.TS

    all_data_list = []

    z_min, z_length, xy_area, nx, ny, nz = get_box_info(fName=logFile)
    eprint(f"XY_Area {xy_area} Z_Length {z_length} A")
    bin_size_z = 0.1
    bin_volume_A3 = xy_area * bin_size_z

    sim_data = read_profile(fName=profileName, bin_volume_A3=bin_volume_A3, max_step_to_read=14000000)
    nAtomsPerBin=nx*ny*n
    vb_data=variable_rebinning(sim_data,xy_area, nAtomsPerBin)
    all_data_list.extend(vb_data)

    all_results = pd.DataFrame(all_data_list)
    dtypes = {
        "timestep": "int",
        "z-coordinate": "float",
        "density": "float",
        "pressure": "float",
        "P_xx": "float",
        "P_yy": "float",
        "P_zz": "float",
        "ke": "float",
        "pe": "float",
        "e_tot": "float",
        "v_com": "float",
        "temperature": "float",
        "n_atoms": "int",
        "zl": "float"
    }
    # os.chdir(wdir)
    all_results = all_results.astype(dtypes)
    # print("All data has been processed and combined into a single DataFrame with correct data types.")
    eprint("All data has been processed and combined into a single DataFrame with correct data types.")


    df_slice=all_results.drop(columns=["n_atoms", "zl", "n_atoms"])

    custom_header = "# direction\tvelocity (m/s)\ttimestep (x 0.00025 fs)\tz-coordinate (A)\tpressure (GPa)\tP_xx (GPa)\tP_yy (GPa)\tP_zz (GPa)\tP_xy (GPa)\tP_xz (GPa)\tP_yz (GPa)\ttau_xz (GPa)\ttau_yz (GPa)\te_tot(eV/atom)\tv_com (m/s)\ttemperature (K)\tdensity (g/cm^3)\n"

    x_column = 'z-coordinate'

    y_columns = ['density', 'temperature', 'P_zz', 'v_com', 'e_tot']  

    y_col_labels = ['Density (g/cm3)', 'Temperature(K)', 'P_zz (GPa)', 'V_z (km/s)', 'E_tot (eV/atom)']

    timesteps = [2000, 4000, 8000, 10000, 20000]
    # timesteps = [2000, 4000]

    for t in timesteps:
        fp = "profile-"+ str(t) + ".dat"
        with open(fp, "w") as f:
            f.write('#Z        Pzz     Temp       E     Rho\n')
            # df_slice = all_results[ (all_results['direction'] == direction) &  (all_results['velocity'] == velocity) & (all_results['timestep'] == t)]
            df_slice = all_results[ (all_results['timestep'] == t) ]
            x2   = np.array(df_slice[x_column])
            pzz2  = np.array(df_slice['P_zz'])
            temp2 = np.array(df_slice['temperature'])
            e2    = np.array(df_slice['pe']+df_slice['ke'])
            rho2  = np.array(df_slice['density'])
            for n in range(len(x2)):
                f.write(f'{x2[n]:.2f} {pzz2[n]:.3f} {temp2[n]:.3f} {e2[n]:5f} {rho2[n]:5f}\n')


if __name__ == "__main__": main(sys.argv[1:])

