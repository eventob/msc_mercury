import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ALL DATA IS NORMALIZED (NON-DIMENSIONALIZED)! Make sure to multiply with the following constants to get the correct units:
mu = 4.0e-7*np.pi           # permeability of free space [H/m] = [T*m/A]
mc = 1.672621777e-27        # mass of a proton [kg]
qc = 1.60217657e-19         # elementary charge [C]
Nc = 3.0e7                  # particles per cubic meter [1/m^3]
Bc = 2.0e-8                 # magnetic field strength [T]
md = 4879.4 / 2             # Mercury's radius [km]
Vc = Bc/np.sqrt(mu*mc*Nc)   # particle velocity [m/s]
Ec = Vc*Bc                  # electric field [V/m]
jc = qc*Nc*Vc               # current density [A/m^2]
fc = qc*Bc / mc             # ion cyclotron frequency [Hz]
xc = Vc / fc                # particle distance [m]
r_m = 2440                  # mercury radius km

def user_input():
    '''
    Function to get user input. The user can choose the following:
        Simulation run
        Cartesian plane
        What parameter to plot
        Time frame / Full run
    '''
    
    # Prompt user to choose sim run and set_plane
    set_run, set_plane = int(input("Simulation run?\n")), str(input("Cartesian set_plane? (eq, mnm, mdd)\n"))
    # Prompt user to choose parameter and time frame
    set_parameter, set_time = int(input("Data set?\n")), str(input("Time step?\n"))
    
    # Reduce computational time by not plotting the grid lines
    if set_time == '00':
        set_hline, set_vline = 0, 0
    else:
        # Prompt user to choose lines to extract data from
        set_hline, set_vline = int(input("Horisontal line? (0-319)\n")), int(input("Vertical line? (0-319)\n"))

    if set_plane == 'eq':     # equatorial set_plane
        set_plane = 'xy'
    elif set_plane == 'mnm':  # meridian set_plane, noon-midnight
        set_plane = 'xz'
    elif set_plane == 'mdd':  # meridian set_plane, dawn-dusk
        set_plane = 'yz'
    
    return set_run, set_plane, set_parameter, set_time, set_hline, set_vline

def get_data(path, index):
    '''
    Function to get data from the .dat-files. The function returns one of the following variables:
    X, Y, N_rep, Nsw, Jx, Jy, Jz, Jex, Jey, Jez, Jrep_x, Jrep_y, Jrep_z, Jsw_x, Jsw_y, Jsw_z,
    Ex, Ey, Ez, Bp_x, Bp_y, Bp_z, Bd_x, Bd_y, Bd_z, BIMF_x, BIMF_y and BIMF_z in that order.
    '''
    file = np.loadtxt(path, skiprows=30)    # Load data from .dat-file
    x, y = file[:, 0]*xc, file[:, 1]*xc     # Extract x- and y-values, normalized to meters
    data = file[:, index]                   # Extract data for the chosen parameter

    # Normalizing the data
    if index == 2 or index == 3:
        data *= (Nc / 1e6)                  # From non-dim to 10^6/m^3
    elif 3 < index < 16:
        data *= jc                  # From non-dim to A/m^2
    elif 15 < index < 19:
        data *= Ec                  # From non-dim to V/m
    elif 18 < index < 28:
        data *= (Bc * 1e9)          # From non-dim to nT
    
    return x, y, data


def name_colorbar(set_parameter):
    '''
    Function to get the correct unit for the plotted colorbar.
    '''
    if set_parameter == 2:
        return '$N_{rep} \\quad [10^6/m^3]$', 'Number density of cold ions - $N_{rep}$'
    elif set_parameter == 3:
        return '$N_{SW} \\quad [10^6/m^3]$', 'Number density of ions - $N_{SW}$'
    elif set_parameter == 4:
        return '$J_x \\quad [A / m^2]$', 'Total current density - $J_x$'
    elif set_parameter == 5:
        return '$J_y \\quad [A / m^2]$', 'Total current density - $J_y$'
    elif set_parameter == 6:
        return '$J_z \\quad [A / m^2]$', 'Total current density - $J_z$'
    elif set_parameter == 7:
        return '$J_{e, x} \\quad [A / m^2]$', 'Electron current density - $J_{e, x}$'
    elif set_parameter == 8:
        return '$J_{e, y} \\quad [A / m^2]$', 'Electron current density - $J_{e, y}$'
    elif set_parameter == 9:
        return '$J_{e, z} \\quad [A / m^2]$', 'Electron current density - $J_{e, z}$'
    elif set_parameter == 10:
        return '$J_{rep, x} \\quad [A / m^2]$', 'Current density of cold ions - $J_{rep, x}$'
    elif set_parameter == 11:
        return '$J_{rep, y} \\quad [A / m^2]$', 'Current density of cold ions - $J_{rep, y}$'
    elif set_parameter == 12:
        return '$J_{rep, z} \\quad [A / m^2]$', 'Current density of cold ions - $J_{rep, z}$'
    elif set_parameter == 13:
        return '$J_{sw, x} \\quad [A / m^2]$', 'Current density of ions - $J_{i, x}$'
    elif set_parameter == 14:
        return '$J_{sw, y} \\quad [A / m^2]$', 'Current density of ions - $J_{i, y}$'
    elif set_parameter == 15:
        return '$J_{sw, z} \\quad [A / m^2]$', 'Current density of ions - $J_{i, z}$'
    elif set_parameter == 16:
        return '$E_x \\quad [V / m]$', 'Electric field - $E_{x}$'
    elif set_parameter == 17:
        return '$E_y \\quad [V / m]$', 'Electric field - $E_{y}$'
    elif set_parameter == 18:
        return '$E_z \\quad [V / m]$', 'Electric field - $E_{z}$'
    elif set_parameter == 19:
        return '$B_{p, x} \\quad [nT]$', 'Induced magnetic field of plasma - $B_{p, x}$'
    elif set_parameter == 20:
        return '$B_{p, y} \\quad [nT]$', 'Induced magnetic field of plasma - $B_{p, y}$'
    elif set_parameter == 21:
        return '$B_{p, z} \\quad [nT]$', 'Induced magnetic field of plasma - $B_{p, z}$'
    elif set_parameter == 22:
        return '$B_{d, x} \\quad [nT]$', 'Mercury dipole magnetic field - $B_{d, x}$'
    elif set_parameter == 23:
        return '$B_{d, y} \\quad [nT]$', 'Mercury dipole magnetic field - $B_{d, y}$'
    elif set_parameter == 24:
        return '$B_{d, z} \\quad [nT]$', 'Mercury dipole magnetic field - $B_{d, z}$'
    elif set_parameter == 25:
        return '$B_{IMF, x} \\quad [nT]$', 'IMF magnetic field - $B_{IMF, x}$'
    elif set_parameter == 26:
        return '$B_{IMF, y} \\quad [nT]$', 'IMF magnetic field - $B_{IMF, y}$'
    elif set_parameter == 27:
        return '$B_{IMF, z} \\quad [nT]$', 'IMF magnetic field - $B_{IMF, z}$'


def satellite_positions():
    '''
    Function to calculate the position of the Mio satellite within the data grid,
    set up by the simulation.
    '''
    # Define the parameters
    N = 33560                                         # number of time steps
    dt = 1                                            # time step [s]
    r_m = 2440 * 1e3                                  # mercury radius [m]
    r_1 = 400 * 1e3                                   # smallest distance from surface [m]
    r_2 = 11824 * 1e3                                 # largest distance from surface [m]
    gravitational_constant = 6.67430e-11              # gravitational constant [m^3 kg^-1 s^-2]
    m_merc = 3.285e23                                 # mass of Mercury [kg]

    semi_major = (r_2 + r_1) / 2 + r_m
    v_sat = np.sqrt(gravitational_constant * m_merc * ((2 / (r_m + r_2)) - (1 / semi_major)))

    # Storage arrays
    pos = np.zeros([N, 2])
    vel = np.zeros([N, 2])

    # Initial conditions
    pos[0, 0] = - r_2 - r_m                              # initial position [m]
    vel[0, 1] = v_sat                                    # initial velocity [m/s], chosen at random

    for i in range(0, N - 1):
        r = np.linalg.norm(pos[i, :])
        acc = (-gravitational_constant * m_merc / r ** 2) * (pos[i, :] / r)
        vel[i + 1, :] = vel[i, :] + acc * dt
        pos[i + 1, :] = pos[i, :] + vel[i + 1, :] * dt
    
    mercury = plt.Circle((0, 0), r_m, color='k', fill=False)
    
    # print(np.rint(pos[17500, :] / 1e3))
    return - pos



def get_figure(data, bar, title, set_yval, set_xval, pos):
    '''
    Function to plot the chosen data. It can plot a single time step or all time steps.
    '''
    x_axis = get_data(path, set_parameter)[0]   # Extracting x-axis values
    y_axis = get_data(path, set_parameter)[1]   # Extracting y-axis values

    X = ((x_axis - (239.625*xc))) / 1e3         # Centering the x-axis around Mercury geocenter
    Y = ((y_axis - (239.625*xc))) / 1e3         # Centering the y-axis around Mercury geocenter
    if set_run == 7:
        X = ((x_axis - ((239.625 - 80)*xc))) / 1e3     # Correcting the x-axis to the new center
    
    pos = pos / 1e3
    mio_traj_data = np.zeros(len(pos))
    tol = 30
    # x_max = max(X)

    for i in range(len(pos)):
        # Take satellite position and look for the closest grid value in the data
        test_area = pos[i, :]
        index_x = np.logical_and(test_area[0] - tol < X, X < test_area[0] + tol)
        index_y = np.logical_and(test_area[1] - tol < Y, Y < test_area[1] + tol)
        one_point = np.logical_and(index_x, index_y)
        if not data[one_point]:
            mio_traj_data[i] = np.nan
        else:
            mio_traj_data[i] = data[one_point][0]


    # Include only the data points that are not zero
    # mio_traj_data = mio_traj_data[mio_traj_data != 0]
    # print(xmio_axis, ymio_axis)

    timestamp_enter = 8675
    timestamp_exit = 33560 - timestamp_enter

    orbit_seconds = len(mio_traj_data[timestamp_enter:timestamp_exit])
    orbit_time = np.linspace(0, orbit_seconds, orbit_seconds)
    # plt.plot(xmio_axis[3000], ymio_axis[3000], 'o', color='black', markersize=1.5, label='Mio start')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', '04:00', '04:50'])
    plt.xlim(-0.1, 9.1)
    plt.xlabel('Time [hours]'), plt.title('Mio simulated data')
    plt.scatter(orbit_time / 1800, mio_traj_data[timestamp_enter:timestamp_exit], s=1.0, linestyle='--', c='black', label='Mio trajectory', zorder=3), plt.grid(color='0.90'), plt.show()
    
    # print(data[one_point], X[index_x], data[index_x].shape, X[index_x].shape)
    
    # plt.plot(X, Y, 'o', color='black', markersize=1.5, label='$grid$'), plt.axis('equal'), plt.show()








set_run, set_plane, set_parameter, set_time, set_yval, set_xval = user_input() # 1, 'xz', 20, 150, 240, 240 # user_input()
path = 'E:/UiO master merkur simulasjoner/mer_r' + str(set_run) + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat'

pos = satellite_positions()
get_figure(get_data(path, set_parameter)[2], name_colorbar(set_parameter)[0], name_colorbar(set_parameter)[1], set_yval, set_xval, pos)

