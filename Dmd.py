# ----------------------------------------------------------- #
#                            DMD                              #
# ----------------------------------------------------------- #

# Author:               Matheus Seidel (matheus.seidel@coc.ufrj.br)
# Revision:             03
# Last update:          17/10/2022

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
General review, formatting, concatenation of pressure and velicty dmd.
'''

# ----------------------------------------------------------- #

from pydmd import DMD
import h5py
import meshio
import os

# ------------------- Parameter inputs ---------------------- #

ti = 10                 # Initial timestep read
tf = 10000              # Final timestep read
par_svd = 16            # SVD rank
par_tlsq = 4            # TLSQ rank
par_exact = True        # Exact (boolean)
par_opt = True          # Opt (boolean)
Pressure_modes = 1      # Run pressure dmd modes? 1 = Y, 0 = N
Pressure_snaps = 1      # Run pressure dmd reconstruction? 1 = Y, 0 = N
Velocity_modes = 1      # Run velocity dmd modes? 1 = Y, 0 = N
Velocity_snaps = 1      # Run velocity dmd reconstruction? 1 = Y, 0 = N

# ------------------------- Data ---------------------------- #

Pressure_data_code = 'f_37'
Velocity_data_code = 'f_31'
Pressure_mesh_path = 'Cilindro_hdmf/Mesh_data_pressure.vtk'
Pressure_data_path = 'Cilindro_hdmf/pressure.h5'
Velocity_mesh_path = 'Cilindro_hdmf/Mesh_data_velocity.vtk'
Velocity_data_path = 'Cilindro_hdmf/velocity.h5'

# ----------------- Reading pressure data ------------------- #

if Pressure_modes == 1 or Pressure_snaps == 1:
    mesh_pressure = meshio.read(Pressure_mesh_path)
    pressure = mesh_pressure.point_data[Pressure_data_code]
    print('Pressure data shape: ', pressure.shape)

    f = h5py.File(Pressure_data_path, 'r')
    data_pressure = f['VisualisationVector']

    snapshots_p = []

    for t in range(ti, tf):
        timestep_p = f[f'VisualisationVector/{t}']
        print(f'Reading time step number {t}')
        snapshots_p.append(timestep_p)

    print(f'{len(snapshots_p)} pressure snapshots were read')
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
    
# ----------------- Reading velocity data ------------------- #

if Velocity_modes == 1 or Velocity_snaps == 1:
    mesh_velocity = meshio.read(Velocity_mesh_path)
    velocity = mesh_velocity.point_data[Velocity_data_code]
    print('Pressure data shape: ', velocity.shape)

    f = h5py.File(Velocity_data_path, 'r')
    data_pressure = f['VisualisationVector']

    snapshots_v = []

    for t in range(ti, tf):
        timestep_v = f[f'VisualisationVector/{t}']
        print(f'Reading time step number {t}')
        snapshots_v.append(timestep_v)

    print(f'{len(snapshots_v)} velocity snapshots were read')
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
    
