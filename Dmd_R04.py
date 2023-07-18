# ----------------------------------------------------------- #
#                            DMD                              #
# ----------------------------------------------------------- #

# Author:               Matheus Seidel (matheus.seidel@coc.ufrj.br)
# Revision:             04
# Last update:          11/07/2023

# Description:
'''  
This code performs the Dynamic Mode Decomposition on fluid simulation data. It uses the PyDMD library to  to generate the DMD modes
and reconstruct the approximation of the original simulation.
The mesh is read by meshio library using vtk files. The simulation data is in h5 format and is read using h5py.
Details about DMD can be found in:
Schmid, P. J., "Dynamic Mode Decomposition of Numerical and Experimental Data". JFM, Vol. 656, Aug. 2010,pp. 5–28. 
doi:10.1017/S0022112010001217
'''

# Last update
'''
Function for reading data.
Error analysis.
'''

# ----------------------------------------------------------- #

from pydmd import DMD
import h5py
import meshio
import os
import numpy as np

# ------------------- Parameter inputs ---------------------- #

ti = 10                 # Initial timestep read
tf = 5000               # Final timestep read
par_svd = 18            # SVD rank
par_tlsq = 18           # TLSQ rank
par_exact = True        # Exact (boolean)
par_opt = True          # Opt (boolean)
Pressure_modes = 1      # Run pressure dmd modes? 1 = Y, 0 = N
Pressure_snaps = 1      # Run pressure dmd reconstruction? 1 = Y, 0 = N
Velocity_modes = 0      # Run velocity dmd modes? 1 = Y, 0 = N
Velocity_snaps = 0      # Run velocity dmd reconstruction? 1 = Y, 0 = N

# ------------------------- Data ---------------------------- #

Pressure_data_code = 'f_26'
Velocity_data_code = 'f_20'
Pressure_mesh_path = 'Cilindro_hdmf/Mesh_data_pressure.vtk'
Pressure_data_path = 'Cilindro_hdmf/solution_p.h5'
Velocity_mesh_path = 'Cilindro_hdmf/Mesh_data_velocity.vtk'
Velocity_data_path = 'Cilindro_hdmf/solution_u.h5'

# ----------------------- Function--------------------------- #

def read_h5_libmesh(filename, dataset):
    """
    Function used to read nodal values from H5 files.

    Parameters
    ----------
    filename : str
        String containing the filename for reading files in h5 (libMesh and EdgeCFD).
    dataset : str
        String containing the dataset desired.

    Returns
    -------
    array : np.array
        Numpy array containing nodal values.
    """
    h5_file = h5py.File(filename, "r")
    data = h5_file[(dataset)]
    data_array = np.array(data, copy=True)
    h5_file.close()
    return data_array

# ----------------- Reading pressure data ------------------- #

current_t = 0 

if Pressure_modes == 1 or Pressure_snaps == 1:
    mesh_pressure = meshio.read(Pressure_mesh_path)
    pressure = mesh_pressure.point_data[Pressure_data_code]
    print('Pressure data shape: ', pressure.shape)

    num_nodes = pressure.shape[0]
    num_time_steps = tf - ti

    snapshots_p = np.zeros((num_nodes, num_time_steps))

    for t in range(ti, tf):
        snapshots = read_h5_libmesh(Pressure_data_path, f'VisualisationVector/{t}')
        print(f'Reading time step number {t}')
        snapshots_p[:, current_t] = snapshots[:, 0]
        current_t = current_t + 1

    print(f'{snapshots_p.shape[1]} pressure snapshots were read')
    print()

# ---------------------- Pressure DMD ----------------------- #

    dmd = DMD(svd_rank=par_svd, tlsq_rank=par_tlsq, exact=par_exact, opt=par_opt)
    dmd.fit(snapshots_p)
    print()
    os.mkdir(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}')

    if Pressure_modes == 1:
        print('DMD modes matrix shape:')
        print(dmd.modes.shape)
        num_modes=dmd.modes.shape[1]
        
        for n in range(0, num_modes):
            print(f'Writing dynamic mode number {n}')
            mode = dmd.modes.real[:, n]
            mesh_pressure.point_data[Pressure_data_code] = mode
            mesh_pressure.write(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}/Mode_{n}.vtk')
        print()

    if Pressure_snaps == 1:
        print('DMD reconstruction matrix shape:')
        print(dmd.reconstructed_data.real.T.shape)

        for t in range(0, tf-ti):
            print(f'Writing dmd timestep number {t}')
            step = dmd.reconstructed_data.real[:, t]
            mesh_pressure.point_data[Pressure_data_code] = step
            mesh_pressure.write(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}/DMD_timestep_{t}.vtk')
        print()

    with open(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}/Pressure_eigs.txt', 'w') as eigs_p:
        eigs_txt = str(dmd.eigs.real)
        eigs_p.write(eigs_txt)

    E = np.linalg.norm(snapshots_p-dmd.reconstructed_data.real, 'fro')/np.linalg.norm(snapshots_p, 'fro')
    E_100 = np.linalg.norm(snapshots_p[:, 89:90]-dmd.reconstructed_data.real[:, 89:90], 'fro')/np.linalg.norm(snapshots_p[:, 89:90], 'fro')
    E_250 = np.linalg.norm(snapshots_p[:, 239:240]-dmd.reconstructed_data.real[:, 239:240], 'fro')/np.linalg.norm(snapshots_p[:, 239:240], 'fro')
    E_5000 = np.linalg.norm(snapshots_p[:, 4989:4990]-dmd.reconstructed_data.real[:, 4989:4990], 'fro')/np.linalg.norm(snapshots_p[:, 4989:4990], 'fro')

    with open(f'Dmd - P_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}/Pressure_error.txt', 'w') as error:
        general_error = str(E)
        error_100 = str(E_100)
        error_250 = str(E_250)
        error_5000 = str(E_5000)
        error.write('General error: ' + general_error + '\n')
        error.write('t=0.10s: ' + error_100 + '\n')
        error.write('t=0.25s: ' + error_250 + '\n')
        error.write('t=5.00s: ' + error_5000 + '\n')

# ----------------- Reading velocity data ------------------- #

current_t = 0 

if Velocity_modes == 1 or Velocity_snaps == 1:
    mesh_velocity = meshio.read(Velocity_mesh_path)
    velocity = mesh_velocity.point_data[Velocity_data_code]
    print('Pressure data shape: ', velocity.shape)

    num_nodes = velocity.shape[0]
    num_time_steps = tf - ti

    snapshots_v = np.zeros((num_nodes, num_time_steps))

    for t in range(ti, tf):
        snapshots = read_h5_libmesh(Velocity_data_path, f'VisualisationVector/{t}')
        print(f'Reading time step number {t}')
        snapshots_v[:, current_t] = snapshots[:, 0]
        current_t = current_t + 1

    print(f'{snapshots_v.shape[1]} pressure snapshots were read')
    print()
# ---------------------- Velocity DMD ----------------------- #

    dmd = DMD(svd_rank=par_svd, tlsq_rank=par_tlsq, exact=par_exact, opt=par_opt)
    dmd.fit(snapshots_v)
    print()
    os.mkdir(f'Dmd - V_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}')

    if Velocity_modes == 1:
        print('DMD modes matrix shape:')
        print(dmd.modes.shape)
        num_modes=dmd.modes.shape[1]
        
        for n in range(0, num_modes):
            print(f'Writing dynamic mode number {n}')
            mode = dmd.modes.real[:, n]
            mesh_velocity.point_data[Velocity_data_code] = mode
            mesh_velocity.write(f'Dmd - V_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}/Mode_{n}.vtk')
        print()

    if Velocity_snaps == 1:
        print('DMD reconstruction matrix shape:')
        print(dmd.reconstructed_data.real.T.shape)

        for t in range(0, tf-ti):
            print(f'Writing dmd timestep number {t}')
            step = dmd.reconstructed_data.real[:, t]
            mesh_velocity.point_data[Pressure_data_code] = step
            mesh_velocity.write(f'Dmd - V_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}/DMD_timestep_{t}.vtk')
        print()

    with open(f'Dmd - V_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}/Velocity_eigs.txt', 'w') as eigs_v:
        eigs_txt = str(dmd.eigs.real)
        eigs_v.write(eigs_txt)

    E = np.linalg.norm(snapshots_v-dmd.reconstructed_data.real, 'fro')/np.linalg.norm(snapshots_v, 'fro')
    E_100 = np.linalg.norm(snapshots_v[:, 89:90]-dmd.reconstructed_data.real[:, 89:90], 'fro')/np.linalg.norm(snapshots_v[:, 89:90], 'fro')
    E_250 = np.linalg.norm(snapshots_v[:, 239:240]-dmd.reconstructed_data.real[:, 239:240], 'fro')/np.linalg.norm(snapshots_v[:, 239:240], 'fro')
    E_5000 = np.linalg.norm(snapshots_v[:, 4989:4990]-dmd.reconstructed_data.real[:, 4989:4990], 'fro')/np.linalg.norm(snapshots_v[:, 4989:4990], 'fro')

    with open(f'Dmd - V_{par_svd}_{par_tlsq}_{par_exact}_{par_opt}/Velocity_error.txt', 'w') as error:
        general_error = str(E)
        error_100 = str(E_100)
        error_250 = str(E_250)
        error_5000 = str(E_5000)
        error.write('General error: ' + general_error + '\n')
        error.write('t=0.10s: ' + error_100 + '\n')
        error.write('t=0.25s: ' + error_250 + '\n')
        error.write('t=5.00s: ' + error_5000 + '\n')    
