import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ALL DATA IS NORMALIZED (NON-DIMENSIONALIZED)! Make sure to multiply with the following constants to get the correct units:
mu = 4.0e-7*np.pi           # permeability of free space [H/m] = [T*m/A]
mc = 1.672621777e-27        # mass of a proton [kg]
qc = 1.60217657e-19         # elementary charge [C]
Nc = 3.0e7                  # particles per cubic meter [1/m^3]
Bc = 2.0e-8                 # interplanetary magnetic field strength [T]
md = 4879.4 / 2             # Mercury's radius [km]
Vc = Bc/np.sqrt(mu*mc*Nc)   # particle velocity [m/s]
Ec = Vc*Bc                  # electric field [V/m]
jc = qc*Nc*Vc               # current density [A/m^2]
fc = qc*Bc / mc             # ion cyclotron frequency [Hz]
xc = Vc / fc                # particle distance [m]
r_m = 2440                  # mercury radius [km]

def user_input():
    '''
    Function to get user input. The user can choose the following:
        Simulation run
        Cartesian plane
        What parameter to plot
        Time frame / Full run
    '''
    # Prompt user to choose sim run and set_plane
    set_run, set_plane = int(input("Simulation run?\n")), str(input("Cartesian set_plane? (eq, mn, dd)\n"))
    # Prompt user to choose parameter and time frame
    set_parameter, set_time = int(input("Data set?\n")), str(input("Time step?\n"))
    if set_time == '00':
        set_lines = 'n'
        set_satellite = 'n'
        print('<----------------------------------------->')
        print("Drawing figures for all timesteps...")
        print('<----------------------------------------->')
    else:
        # Prompt user to send mio through data
        set_satellite = str(input("Throw Mio into the environment? (y/n)\n"))
        # Prompt user to choose lines or not
        set_lines = str(input("Extract data from lines? (y/n)\n"))
        # Prompt user to set colorbar limits
        set_limits = input("Set colorbar limits? (y/n)\n")

    # Reduce computational time by not plotting the grid lines
    if set_lines == 'y':
        # Prompt user to choose lines to extract data from
        set_hline, set_vline = int(input("Horisontal line? (0-319)\n")), int(input("Vertical line? (0-319)\n"))
    else:
        # Bypass the if-tests by setting the lines to 0
        set_hline, set_vline = 0, 0
    if set_plane == 'eq':
        # eq = equatorial plane
        set_plane = 'xy'
    elif set_plane == 'mn':
        # mn = meridian midnight-noon plane
        set_plane = 'xz'
    elif set_plane == 'dd':
        # dd = meridian dawn-dusk plane
        set_plane = 'yz'
    if set_satellite == 'y':
        # Prompt user to choose the rotation of the satellite orbit
        set_rot = int(input("Choose the TAA of the satellite orbit (0, 90, 180, 270): "))

    return set_run, set_plane, set_parameter, set_time, set_hline, set_vline, set_satellite, set_limits, set_rot

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


def satellite_position():
    '''
    Function to calculate the position of the Mio satellite within the data grid,
    set up by the simulation. Mercury is placed in the center with (x, y) = (0, 0).
    '''
    

    # Define the parameters
    N = 33560                                       # number of timesteps for one full orbit
    dt = 1                                          # timestep [s]
    gravitational_constant = 6.67430e-11            # gravitational constant [m^3 kg^-1 s^-2]
    m_merc = 3.285e23                               # mass of Mercury [kg]
    r_merc = r_m * 1e3                              # mercury radius [m]
    r_1 = 400 * 1e3                                 # smallest distance from surface, periapsis [m]
    r_2 = 11824 * 1e3                               # largest distance from surface, apoapsis [m]
    semi_major = (2 * r_merc + r_2 + r_1) / 2       # semi-major axis [m]
    a = r_merc + r_2                                # distance from focus to apsis, where the center of Mercury is the focus [m]
    timestamp_enter = 8675                          # timestamp where the satellite enters the simulation data grid (after 8675 sec)
    timestamp_exit = N - timestamp_enter            # timestamp where the satellite exits the simulation data grid (after 33560 - 8675 sec)

    # Initial satellite velocity for an elliptical orbit
    v_sat = np.sqrt(gravitational_constant * m_merc * ((2 / a) - (1 / semi_major)))

    # Satellite position and velocity arrays in 2D
    pos = np.zeros([N, 2])
    vel = np.zeros([N, 2])

    # Initial position and velocity based on the TAA of Mio's orbit
    if set_rot == 0:
        # From solar wind direction, inside nightside tail
        pos[0, 0] = - r_2 - r_merc
        vel[0, 1] = - v_sat
    elif set_rot == 90:
        # "Above" planet, outside bowshock to inside magnetopause
        pos[0, 1] = r_2 + r_merc
        vel[0, 0] = - v_sat
    elif set_rot == 180:
        # From tail, inside magnetopause
        pos[0, 0] = r_2 + r_merc
        vel[0, 1] = v_sat
    
    elif set_rot == 270:
        # "Below" planet, inside bowshock to outside magnetopause
        pos[0, 1] = - r_2 - r_merc
        vel[0, 0] = v_sat

    # Extract the grid data for the virtual Mio data
    for i in range(0, N - 1):
        # Update the acceleration, velocity and position of the satellite using the Euler-Cromer method
        radius = np.linalg.norm(pos[i, :])
        acc = (-gravitational_constant * m_merc / radius ** 2) * (pos[i, :] / radius)
        vel[i + 1, :] = vel[i, :] + acc * dt
        pos[i + 1, :] = pos[i, :] + vel[i + 1, :] * dt

    # Extract the satellite positions within the data grid and convert to km
    inside_datagrid = pos[timestamp_enter:timestamp_exit] / 1e3

    # Draw the satellite's orbit with time markers
    min_to_sec = 1800
    plt.plot(inside_datagrid[:, 0], inside_datagrid[:, 1], label="TAA = " + str(set_rot) + '${}^\\circ$')
    plt.plot(inside_datagrid[0 * min_to_sec, 0], inside_datagrid[0 * min_to_sec, 1], 'o', color='limegreen', markersize=3)
    for i in range(1, 10):
        # Mark every 30 minutes with a red dot
        plt.plot(inside_datagrid[i * min_to_sec, 0], inside_datagrid[i * min_to_sec, 1], 'ro', markersize=3)
    
    return inside_datagrid, min_to_sec


def satellite_data(sat_coord, X, Y, set_parameter, set_rot, conv_min):
    '''
    Function to plot and extract Mio's virtual data along the trajectory.
    '''
    # Draw a plot with Mio's virtual data along the trajectory
    fig, ax = plt.subplots(5, figsize=(8, 12), sharex=True, sharey=False, layout='tight')
    fig.suptitle("Electric field and number density along Mio's orbit at TAA = " + str(set_rot) + '${}^\\circ$', fontsize=16)
    # fig.supxlabel('Time [hours]'), fig.supylabel(name_colorbar(set_parameter)[0])
    sat_time = np.linspace(0, len(sat_coord), len(sat_coord))
    sat_time = sat_time / conv_min

    # Retrieve satellite data along Mio trajectory and mark every 30 minutes
    grid_tol = 26                                                # Tolerance for the satellite data (grid distance in km; 52 km between each grid point)
    
    plotnum = 0
    paths = 'E:/UiO master merkur simulasjoner/mer_r1' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            'E:/UiO master merkur simulasjoner/mer_r6' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            'E:/UiO master merkur simulasjoner/mer_r5' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            'E:/UiO master merkur simulasjoner/mer_r8' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            'E:/UiO master merkur simulasjoner/mer_r9' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat'
            # 'E:/UiO master merkur simulasjoner/mer_r2' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            # 'E:/UiO master merkur simulasjoner/mer_r3' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            # 'E:/UiO master merkur simulasjoner/mer_r4' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            # 'E:/UiO master merkur simulasjoner/mer_r7' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
    
    sat_xdata = np.zeros(len(sat_coord))                          # Empty array for the x-component satellite data
    sat_ydata = sat_xdata.copy()                                  # Empty array for the y-component satellite data
    sat_zdata = sat_xdata.copy()                                  # Empty array for the z-component satellite data
    num_dens = sat_xdata.copy()                                   # Empty array for the number density data
    for j in paths:
        # data_x = get_data(j, set_parameter)[2] + get_data(j, set_parameter + 3)[2] + get_data(j, set_parameter + 6)[2]               # Extract the data for the magnetic field
        # data_y = get_data(j, set_parameter + 1)[2] + get_data(j, set_parameter + 4)[2] + get_data(j, set_parameter + 7)[2]           # Extract the data for the magnetic field
        # data_z = get_data(j, set_parameter + 2)[2] + get_data(j, set_parameter + 5)[2] + get_data(j, set_parameter + 8)[2]           # Extract the data for the magnetic field
        
        data_x = get_data(j, set_parameter)[2]          # Extract the data for electric field
        data_y = get_data(j, set_parameter + 1)[2]      # Extract the data for electric field
        data_z = get_data(j, set_parameter + 2)[2]      # Extract the data for electric field
        
        number_density = get_data(j, 3)[2]

        # data = get_data(i, 20)[2] + get_data(i, 21)[2]               # Extract the measured magnetic field data Bd + Bp

    
        for i in range(len(sat_coord)):
            # Take the satellite position values and look for the closest grid value in the data
            test_pos = sat_coord[i, :]
            index_x = np.logical_and(test_pos[0] - grid_tol < X, X < test_pos[0] + grid_tol)
            index_y = np.logical_and(test_pos[1] - grid_tol < Y, Y < test_pos[1] + grid_tol)
            xy_check = np.logical_and(index_x, index_y)
            if data_x[xy_check].shape == (0,):
                sat_xdata[i] = np.nan
                sat_ydata[i] = np.nan
                sat_zdata[i] = np.nan
                num_dens[i] = np.nan
            else:
                sat_xdata[i] = data_x[xy_check][0]
                sat_ydata[i] = data_y[xy_check][0]
                sat_zdata[i] = data_z[xy_check][0]
                num_dens[i] = number_density[xy_check][0]
        
        sat_xdata = pd.Series(sat_xdata).interpolate().values          # Interpolate the missing data values
        sat_ydata = pd.Series(sat_ydata).interpolate().values          # Interpolate the missing data values
        sat_zdata = pd.Series(sat_zdata).interpolate().values          # Interpolate the missing data values
        num_dens = pd.Series(num_dens).interpolate().values            # Interpolate the missing data values
        tot_data = np.sqrt(sat_xdata ** 2 + sat_ydata ** 2 + sat_zdata ** 2)

        # plt.scatter(sat_time / 1800, sat_data, c=sat_data, cmap='plasma', s=1, zorder=4)
        names = 'S1', 'S6', 'S5', 'S8', 'S9'
        color = 'black', 'red', 'blue', 'green', 'purple'
        # ax[plotnum].axvline(8104 / conv_min, label='Mercury geological equator', linestyle='--', color='0.85')
        ax[0].plot(sat_time, sat_xdata, color=color[plotnum], zorder=4, label=names[plotnum])
        ax[0].set_ylabel('$\\mathbf{E}_x$ [nT]')
        ax[1].plot(sat_time, sat_ydata, color=color[plotnum], zorder=4, label=names[plotnum])
        ax[1].set_ylabel('$\\mathbf{E}_y$ [nT]')
        ax[2].plot(sat_time, sat_zdata, color=color[plotnum], zorder=4, label=names[plotnum])
        ax[2].set_ylabel('$\\mathbf{E}_z$ [nT]')
        ax[3].plot(sat_time, tot_data, color=color[plotnum], zorder=4, label=names[plotnum])
        ax[3].set_ylabel('$|\\mathbf{E}|$ [nT]')
        ax[4].plot(sat_time, num_dens, color=color[plotnum], zorder=4, label=names[plotnum])
        ax[4].set_ylabel('$N_{SW}$ [$10^6/m^3$]')
        ax[plotnum].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', '04:00', '04:30'])
        ax[plotnum].grid(color='0.8') # ax[plotnum].set_xlim(-0.1, 9.1), 
        plotnum += 1
    
    ax[0].legend(loc='best', fontsize='small')
    ax[1].legend(loc='best', fontsize='small')
    ax[2].legend(loc='best', fontsize='small')
    ax[3].legend(loc='best', fontsize='small')
    ax[4].legend(loc='best', fontsize='small')
    plt.savefig('E:/UiO master merkur simulasjoner/animation/plots/' + 'mio_data_temp' + '.pdf', bbox_inches='tight')
    plt.clf(), plt.cla(), plt.close()


def draw_2d_figure(path, bar, title, set_yval, set_xval):
    '''
    Function to draw the chosen data into a 2D grid.
    It has the option to draw a horisontal and vertical line at a specified position.
    In addition to Mio's satellite orbit.
    '''
    # Extract the spatial- and parameter data from the .dat-file
    fig, ax = plt.subplots(1, 1)
    plt.figure(1, layout='tight')
    x_grid, y_grid, data = get_data(path, set_parameter)

    # Move the center of the simulation to Mercury's geocenter and convert to km
    X = ((x_grid - (239.625*xc))) / 1e3
    Y = ((y_grid - (239.625*xc))) / 1e3
    
    # Conditions dependent on user input
    if set_run == 7:
        # Adjust the center of the simulation if visualizing simulation 7
        X = ((x_grid - ((239.625 - 80)*xc))) / 1e3
    if set_satellite == 'y':
        # Draw Mio's satellite orbit in 2D
        sat_pos, conv_min = satellite_position()
        satellite_data(sat_pos, X, Y, set_parameter, set_rot, conv_min)
    if set_limits == 'y':
        # Set the colorbar limits manually
        data_min = input("Min value: ")
        if data_min == 'neg':
            data_min = - float(input("Min negative value: "))
        else:
            data_min = float(data_min)
        data_max = float(input("Max value: "))
        levels = np.linspace(data_min, data_max, 20)
    else:
        # No input from user, set colorbar to min and max of data
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        levels = np.linspace(data_min, data_max, 20)

    # Draw the grid data in 2D
    para_grid = ax.tricontourf(X, Y, data, cmap='plasma', levels=levels, antialiased=False)
    fig.colorbar(para_grid, label=str(bar))
    
    # Draw the Mercury projection and geocenter
    m_radius = r_m
    m_proj = plt.Circle((0, 0), m_radius, color='black', fill=False, zorder=2, linewidth=0.8, label='$S_M$')
    ax.add_patch(m_proj)
    ax.plot(0, 0, 'o', color='black', markersize=1.5, label='$C_p$')

    # Figure settings for the 2D grid
    ax.axis('equal'), ax.grid(alpha=0.8, linestyle="--", color='gray'), ax.set_title(str(title))
    ax.set_xticks(np.array([-4 * r_m, -3 * r_m, -2 * r_m, -1 * r_m, 0, 1 * r_m, 2 * r_m, 3 * r_m, 4 * r_m]), [-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax.set_yticks(np.array([-4 * r_m, -3 * r_m, -2 * r_m, -1 * r_m, 0, 1 * r_m, 2 * r_m, 3 * r_m, 4 * r_m]), [-4, -3, -2, -1, 0, 1, 2, 3, 4])
    if set_plane == 'xy':
        # Change labels to xy
        ax.set_xlabel('x / $R_M$'), ax.set_ylabel('y / $R_M$')
    elif set_plane == 'xz':
        # Change labels to xz and mark Mercury's magnetic center offset
        ax.set_xlabel('x / $R_M$'), ax.set_ylabel('z / $R_M$')
        ax.plot(0, 11.736*xc / 1e3, 'o', color='red', markersize=1.5, label='$C_m$')    # Mark Mercury's magnetic center in the xz-plane

    # Draw the induced magnetic field vectors over the simulation data
    # ax.quiver(X[::250], Y[::250], get_data(path, 19)[2][::250], get_data(path, 20)[2][::250], color='black', zorder=3, label='Bp')
    
    # Draw an horizontal line at a chosen y-value over the 2D grid
    if set_yval != 0:
        # Only if the chosen y-value is different than 0, search the grid for the corresponding x-value
        y_corr = np.array([Y[0 + (320 * i)] for i in range(0, 320)])       
        ax.hlines(y_corr[set_yval], min(X), 0, color='lightgreen', zorder=3, linewidth=1, linestyle='--', label='HL')
    
    # Draw an vertical line at a chosen x-value over the 2D grid
    if set_xval != 0:
        # Only if the chosen x-value is different than 0, search the grid for the corresponding y-value
        x_corr = np.array(X[0:319])                    # Correct x-axis values to input
        ax.vlines(x_corr[set_xval], min(Y), max(Y), color='lightgreen', zorder=3, linewidth=1, linestyle='--', label='VL')      # Plotting the vertical line
    
    # Save the figure and close the figure window
    plt.legend(loc='best', fontsize='small')
    plt.savefig('E:/UiO master merkur simulasjoner/animation/plots/' + 'sim_data_temp' + '.png', bbox_inches='tight', dpi=1000)
    plt.clf()

    
def linear_data_extraction(set_xval, set_yval, index):
    """
    Function for extracting data from the .dat-files for a linear analysis.
    """

    
    paths = 'E:/UiO master merkur simulasjoner/mer_r1' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            'E:/UiO master merkur simulasjoner/mer_r6' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            'E:/UiO master merkur simulasjoner/mer_r5' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            'E:/UiO master merkur simulasjoner/mer_r8' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            'E:/UiO master merkur simulasjoner/mer_r9' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat'
            # 'E:/UiO master merkur simulasjoner/mer_r2' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            # 'E:/UiO master merkur simulasjoner/mer_r3' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            # 'E:/UiO master merkur simulasjoner/mer_r4' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            # 'E:/UiO master merkur simulasjoner/mer_r7' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
    
    # paths = 'E:/UiO master merkur simulasjoner/mer_r1' + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat', \
            
    x_ax = get_data(paths[0], index)[0]                             # Extract x-axis data values
    draw_xax = np.array(((x_ax[0:320] - (239.625*xc))) / 1e3)       # Draw x-axis corrected for the center of Mercury
    # Change the x- and y-axis labels depending on the chosen plane
    if set_plane == 'xy':
        coordinates = '0, ' + '%.2f' % (int(draw_xax[set_yval]) / r_m) + ' , 0) $R_M$'
    elif set_plane == 'xz':
        coordinates = '0, 0, ' + '%.2f' % (int(draw_xax[set_yval]) / r_m)
    plotnum = 1
    fig, ax = plt.subplots(5, figsize=(8, 12), sharex=True, sharey=True, layout='tight')
    fig.suptitle(name_colorbar(set_parameter)[1] + ' along HL at (' + coordinates + ') $R_M$', fontsize=16)
    # fig.set_ylabel(name_colorbar(set_parameter)[0])
    # plt.ylabel()
    for i in paths:
        data = get_data(i, index)[2]                                                            # Extracting the data for the chosen parameter
        hdata_line = data[(0 + (320 * set_yval)):(320 + (320 * set_yval))]                      # Fetch the line of data at the chosen y-value
        vdata_line = [data[0 + (set_xval + (320 * i))] for i in range(0, 320)]                  # Fetch one value from each line of data at all y-values for a chosen x-value
        
        # Find the bow shock within the data
        # 0.001 for cold background ions, 0.1 for anything else
        tolerance = 0.001 * max(abs(hdata_line))           # Tolerance of 10 % of the max value of the data
        for j in range(25, 319):
            difference = abs(hdata_line[j] - hdata_line[j - 1])
            if difference > tolerance:
                bs_pos = draw_xax[j - 1]
                print(bs_pos / r_m, j, 1)
                break
        
        # Find the magnetopause within the data, by educated guess
        # Number density guesses = 1: 88, 2: 94, 3: 91, 4: 91, 5: 86
        # Current density guesses = 1: 86, 2: 93, 3: 90, 4: 89, 5: 85
        # Cold ion background guesses = Difficult to find
        
        if plotnum == 1:
            mp_pos = draw_xax[86]
            print(mp_pos / r_m, 2)
        elif plotnum == 2:
            mp_pos = draw_xax[93]
            print(mp_pos / r_m, 2)
        elif plotnum == 3:
            mp_pos = draw_xax[90]
            print(mp_pos / r_m, 2)
        elif plotnum == 4:
            mp_pos = draw_xax[89]
            print(mp_pos / r_m, 2)
        elif plotnum == 5:
            mp_pos = draw_xax[85]
            print(mp_pos / r_m, 2)
        
        # tolerance_mp = 0.1 * max(abs(hdata_line))           # Tolerance of 10 % of the max value of the data
        # for k in range(1, 319):
        #     hdata_line_rev = np.flip(hdata_line)
        #     difference_mp = abs(hdata_line_rev[k] - hdata_line_rev[k - 1])
        #     # print(difference, tolerance)
        #     if difference_mp > tolerance_mp:
        #         mp_pos = draw_xax[319 - k - 1]
        #         print(mp_pos / r_m, k, 2)
        #         break
        print(draw_xax[122] / r_m, 3)
        # Data extracting and plotting scheme for a horisontal line at a chosen value of y
        if set_yval != 0:
            # plt.subplot(5, 1, plotnum)
            # if plotnum == 1:
                # plt.title(name_colorbar(set_parameter)[1] + ' along HL at (' + coordinates)
            names = 'S1', 'S6', 'S5', 'S8', 'S9'
            ax[plotnum - 1].plot(draw_xax[0:159], hdata_line[0:159], color='k', zorder=2, label=names[plotnum - 1])
            # plt.scatter(draw_xax, hdata_line, label='HL data', s=2, color='k', zorder=1)
            ax[plotnum - 1].axvline(x=bs_pos, color='orange', linestyle='--', alpha=1, zorder=0, label='$R_{BS}$')
            # ax[plotnum - 1].axvline(x=mp_pos, color='pink', linestyle='--', alpha=1, zorder=0, label='$R_{MP}$')
            ax[plotnum - 1].axvline(x=draw_xax[122], color='gray', linestyle='--', alpha=1, zorder=3, label='$S_M$')    # Mercury's surface at z = 0.2, -0.95 R_M
            ax[plotnum - 1].grid(zorder=0)
            ax[plotnum - 1].set_xticks(np.array([-4 * r_m, -3 * r_m, -2 * r_m, -1 * r_m, 0]), [-4, -3, -2, -1, 0])# ([-4 * r_m, -3 * r_m, -2 * r_m, -1 * r_m, 0, 1 * r_m, 2 * r_m, 3 * r_m, 4 * r_m]), [-4, -3, -2, -1, 0, 1, 2, 3, 4])  # Setting up the x-axis ticks
            ax[4].set_xlabel('x / $R_M$', size=12)
            ax[2].set_ylabel(name_colorbar(set_parameter)[0], size=12)
            ax[plotnum - 1].legend(loc='best', fontsize='small')
        plotnum += 1

    # plt.legend(loc='best', fontsize='small')
    plt.savefig('E:/UiO master merkur simulasjoner/animation/plots/' + 'HL_temp_fig' + '.pdf')
        
    # Plotting scheme for a vertical line at a chosen value of x
    if set_xval != 0:
        # print("Data extracted at x = " + str(draw_xax[set_xval]) + " km")
        plt.figure(3)
        plt.scatter(draw_xax, vdata_line, label='VL data', s=2, color='k', zorder=1)
        if set_plane == 'xy':
            plt.xlabel('y / $R_M$')
            coordinates = '%.2f' % (int(draw_xax[set_xval]) / r_m) + ', 0, 0) $R_M$'
        elif set_plane == 'xz':
            plt.xlabel('z / $R_M$')
            coordinates = '%.2f' % (int(draw_xax[set_xval]) / r_m) + ', 0, 0) $R_M$'
        elif set_plane == 'yz':
            plt.xlabel('z / $R_M$')
            coordinates = '(0, ' + '%.2f' % (int(draw_xax[set_xval]) / r_m) + ' , 0) $R_M$'
        # plt.scatter([draw_xax[152], draw_xax[166]], [vdata_line[152], vdata_line[166]], color = 'r', s=10, label='Mio crossings', zorder=2)
        plt.ylabel(name_colorbar(set_parameter)[0]), plt.title(name_colorbar(set_parameter)[1] + ' along VL at (' + coordinates)
        plt.xticks(np.array([-4 * r_m, -3 * r_m, -2 * r_m, -1 * r_m, 0, 1 * r_m, 2 * r_m, 3 * r_m, 4 * r_m]), [-4, -3, -2, -1, 0, 1, 2, 3, 4])  # Setting up the x-axis ticks
        plt.grid(), plt.legend(loc='best', fontsize='small')
        plt.savefig('E:/UiO master merkur simulasjoner/animation/plots/' + 'VL_temp_fig' + '.pdf')


'''
    N_rep = 2, Nsw = 3, Jx = 4, Jy = 5, Jz = 6, Jex = 7, Jey = 8, Jez = 9, 
    Jrep_x = 10, Jrep_y = 11, Jrep_z = 12, Jsw_x = 13, Jsw_y = 14, Jsw_z = 15,
    Ex = 16, Ey = 17, Ez = 18, Bp_x = 19, Bp_y = 20, Bp_z = 21, 
    Bd_x = 22, Bd_y = 23, Bd_z = 24, BIMF_x = 25, BIMF_y = 26, BIMF_z = 27.
'''
set_run, set_plane, set_parameter, set_time, set_yval, set_xval, set_satellite, set_limits, set_rot = user_input() # 1, 'xz', 20, 150, 240, 240 # user_input()
path = 'E:/UiO master merkur simulasjoner/mer_r' + str(set_run) + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(set_time) + '.dat'


if set_time == '00':
    timesteps = np.array(['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', \
                          '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', \
                          '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', \
                          '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', \
                          '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', \
                          '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', \
                          '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', \
                          '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', \
                          '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', \
                          '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', \
                          '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', \
                          '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', \
                          '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', \
                          '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', \
                          '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', \
                          '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', \
                          '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', \
                          '171', '172', '173', '174', '175', '176', '177', '178', '179', '180'])
    for i in timesteps:
        path = 'E:/UiO master merkur simulasjoner/mer_r' + str(set_run) + '/tec2D/2D_' + str(set_plane) + '/prop' + str(set_plane) + '00' + str(i) + '.dat'
        print('Making figure ' + str(i))
        print(path)
        if Path(path).is_file() == False:
            print('<----------------------------------------->')
            print('No data for timeframe ' + str(i) + ' found.')
            print('We have reached the bottom. No more files to read.')
            print('Exiting...')
            print('<----------------------------------------->')
            break
        else:
            draw_2d_figure(path, name_colorbar(set_parameter)[0], name_colorbar(set_parameter)[1], set_yval, set_xval)
            plt.legend(loc='best', fontsize='small')
            plt.savefig('E:/UiO master merkur simulasjoner/animation/plots/mer_r' + str(set_run) + '/mer_r' + str(set_run) + '_' + str(set_parameter) + '_' + str(set_plane) + '_' + str(i) + '.png')
            plt.close()
            print('Figure ' + str(i) + ' done.')
            print('<----------------------------------------->')
else:
    draw_2d_figure(path, name_colorbar(set_parameter)[0], name_colorbar(set_parameter)[1], set_yval, set_xval)
    # linear_data_extraction(set_xval, set_yval, set_parameter)
    # elliptical()
    # plt.show()
