import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ALL DATA IS NORMALIZED (NON-DIMENSIONALIZED)! Make sure to multiply with the following constants to get the correct units:
mu = 4.0e-7*np.pi           # permeability of free space [H/m] = [T*m/A]
mc = 1.672621777e-27        # mass of a proton [kg]
qc = 1.60217657e-19         # elementary charge [C]
Nc = 3.0e7                  # particles per cubic meter [1/m^3]
Bc = 2.0e-8                 # magnetic field strength [T]
md = 4879.4 / 2                 # Mercury's radius [km]
Vc = Bc/np.sqrt(mu*mc*Nc)   # particle velocity [m/s]
Ec = Vc*Bc                  # electric field [V/m]
jc = qc*Nc*Vc               # current density [A/m^2]
fc = qc*Bc / mc             # particle frequency [Hz]
xc = Vc / fc                # particle distance [m]

def user_input():
    '''
    Function to get user input. The user can choose to run the simulation or not, and choose which plane to plot.
    '''
    usr1, usr3 = int(input("Simulation run?\n")), str(input("Cartesian plane? (eq, mnm, mdd)\n"))
    usr2, usr4 = str(input("Time step?\n")), int(input("Data set?\n"))

    if usr3 == 'eq':     # equatorial plane
        usr3 = 'xy'
    elif usr3 == 'mnm':  # meridian plane, noon-midnight
        usr3 = 'xz'
    elif usr3 == 'mdd':  # meridian plane, dawn-dusk
        usr3 = 'yz'
    
    return usr1, usr2, usr3, usr4

def get_data(path):
    '''
    Function to get data from the .dat-files. The function returns the following variables:
    X, Y, N_rep, Nsw, Jx, Jy, Jz, Jex, Jey, Jez, Jrep_x, Jrep_y, Jrep_z, Jsw_x, Jsw_y, Jsw_z,
    Ex, Ey, Ez, Bp_x, Bp_y, Bp_z, Bd_x, Bd_y, Bd_z, BIMF_x, BIMF_y and BIMF_z in that order.
    '''
    # print(path)
    file = np.loadtxt(path, skiprows=30)
    # print(file[:, 0]*xc)
    x, y, z = file[:, 0]*xc, file[:, 1]*xc, file[:, 2]*xc
    N_rep, Nsw = file[:, 2]*Nc, file[:, 3]*Nc
    Jx, Jy, Jz = file[:, 4]*jc, file[:, 5]*jc, file[:, 6]*jc
    Jex, Jey, Jez = file[:, 7]*jc, file[:, 8]*jc, file[:, 9]*jc
    Jrep_x, Jrep_y, Jrep_z = file[:, 10]*jc, file[:, 11]*jc, file[:, 12]*jc
    Jsw_x, Jsw_y, Jsw_z = file[:, 13]*jc, file[:, 14]*jc, file[:, 15]*jc
    Ex, Ey, Ez = file[:, 16]*Ec, file[:, 17]*Ec, file[:, 18]*Ec
    Bp_x, Bp_y, Bp_z = file[:, 19]*Bc, file[:, 20]*Bc, file[:, 21]*Bc
    Bd_x, Bd_y, Bd_z = file[:, 22]*Bc, file[:, 23]*Bc, file[:, 24]*Bc
    BIMF_x, BIMF_y, BIMF_z = file[:, 25]*Bc, file[:, 26]*Bc, file[:, 27]*Bc
    
    # Only syntax-values from 3 to 28 are valid.
    return x, y, z, N_rep, Nsw, Jx, Jy, Jz, Jex, Jey, Jez, Jrep_x, Jrep_y, Jrep_z, \
           Jsw_x, Jsw_y, Jsw_z, Ex, Ey, Ez, Bp_x, Bp_y, Bp_z, Bd_x, Bd_y, Bd_z, BIMF_x, BIMF_y, BIMF_z

'''
Values are returned as strings.
    N_rep = 3, Nsw = 4, Jx = 5, Jy = 6, Jz = 7, Jex = 8, Jey = 9, Jez = 10, 
    Jrep_x = 11, Jrep_y = 12, Jrep_z = 13, Jsw_x = 14, Jsw_y = 15, Jsw_z = 16,
    Ex = 17, Ey = 18, Ez = 19, Bp_x = 20, Bp_y = 21, Bp_z = 22, 
    Bd_x = 23, Bd_y = 24, Bd_z = 25, BIMF_x = 26, BIMF_y = 27, BIMF_z = 28.
'''
def name_colorbar(usr4):
    '''
    Function to get the correct name of the plotted colorbar.
    '''
    if usr4 == 3:
        return '$N_{rep} \\quad [1/m^3]$', 'Number density of cold ions'
    elif usr4 == 4:
        return '$N_{sw} \\quad [1/m^3]$', 'Number density of ions'
    elif usr4 == 5:
        return '$J_x \\quad [A / m^2]$', 'Total current density (x-component)'
    elif usr4 == 6:
        return '$J_y \\quad [A / m^2]$', 'Total current density (y-component)'
    elif usr4 == 7:
        return '$J_z \\quad [A / m^2]$', 'Total current density (z-component)'
    elif usr4 == 8:
        return '$J_{e, x} \\quad [A / m^2]$', 'Electron current density (x-component)'
    elif usr4 == 9:
        return '$J_{e, y} \\quad [A / m^2]$', 'Electron current density (y-component)'
    elif usr4 == 10:
        return '$J_{e, z} \\quad [A / m^2]$', 'Electron current density (z-component)'
    elif usr4 == 11:
        return '$J_{rep, x} \\quad [A / m^2]$', 'Current density of cold ions (x-component)'
    elif usr4 == 12:
        return '$J_{rep, y} \\quad [A / m^2]$', 'Current density of cold ions (y-component)'
    elif usr4 == 13:
        return '$J_{rep, z} \\quad [A / m^2]$', 'Current density of cold ions (z-component)'
    elif usr4 == 14:
        return '$J_{sw, x} \\quad [A / m^2]$', 'Current density of ions (x-component)'
    elif usr4 == 15:
        return '$J_{sw, y} \\quad [A / m^2]$', 'Current density of ions (y-component)'
    elif usr4 == 16:
        return '$J_{sw, z} \\quad [A / m^2]$', 'Current density of ions (z-component)'
    elif usr4 == 17:
        return '$E_x \\quad [V / m]$', 'Electric field (x-component)'
    elif usr4 == 18:
        return '$E_y \\quad [V / m]$', 'Electric field (y-component)'
    elif usr4 == 19:
        return '$E_z \\quad [V / m]$', 'Electric field (z-component)'
    elif usr4 == 20:
        return '$B_{p, x} \\quad [T]$', 'Induced magnetic field of plasma (x-component)'
    elif usr4 == 21:
        return '$B_{p, y} \\quad [T]$', 'Induced magnetic field of plasma (y-component)'
    elif usr4 == 22:
        return '$B_{p, z} \\quad [T]$', 'Induced magnetic field of plasma (z-component)'
    elif usr4 == 23:
        return '$B_{d, x} \\quad [T]$', 'Dipole magnetic field (x-component)'
    elif usr4 == 24:
        return '$B_{d, y} \\quad [T]$', 'Dipole magnetic field (y-component)'
    elif usr4 == 25:
        return '$B_{d, z} \\quad [T]$', 'Dipole magnetic field (z-component)'
    elif usr4 == 26:
        return '$B_{IMF, x} \\quad [T]$', 'IMF magnetic field (x-component)'
    elif usr4 == 27:
        return '$B_{IMF, y} \\quad [T]$', 'IMF magnetic field (y-component)'
    elif usr4 == 28:
        return '$B_{IMF, z} \\quad [T]$', 'IMF magnetic field (z-component)'

def get_figure(data, bar, title):
    '''
    Function to plot the chosen data. It can plot a single time step or all time steps.
    '''
    X = ((get_data(path)[0] - (239.625*xc))) / 1e3
    Y = ((get_data(path)[1] - (239.625*xc))) / 1e3
    merc_proj = plt.Circle((0, 0), 58.683*xc / 1e3, color='black', fill=False, zorder=2, linewidth=0.5)

    fig, ax = plt.subplots(1, 1)
    merc_xy = ax.tricontourf(X, Y, data, cmap='plasma') # , vmin=1e8, vmax=2e8)
    # horisontal_line = plt.hlines(0, -10000, 10000, color='black', zorder=3, linewidth=0.5)
    # magnetopause_tip = ax.vlines(-3662.44, linestyle='--', color='black', ymin=-10000, ymax=10000, zorder=3, linewidth=0.5)
    # ax.quiver(X, Y, BIMF_x, BIMF_z, color='black', zorder=3, linewidth=0.5, scale=1e-3, scale_units='inches')
    fig.colorbar(merc_xy, label=str(bar))
    ax.add_patch(merc_proj)
    ax.axis('equal'), ax.grid(alpha=0.4, linestyle="--"), ax.set_title(str(title))
    if usr3 == 'xy':
        ax.set_xlabel('x / $R_M$'), ax.set_ylabel('y / $R_M$')
    elif usr3 == 'xz':
        ax.set_xlabel('x / $R_M$'), ax.set_ylabel('z / $R_M$')
    elif usr3 == 'yz':
        ax.set_xlabel('y / $R_M$'), ax.set_ylabel('z / $R_M$')
    # plt.xticks([-4*md, -3*md, -2*md, -md, 0, md, 2*md, 3*md, 4*md], [-4, -3, -2, 1, 0, 1, 2, 3, 4]), plt.yticks([-4*md, -3*md, -2*md, -md, 0, md, 2*md, 3*md, 4*md], [-4, -3, -2, 1, 0, 1, 2, 3, 4])
    if usr2 == '00':
        plt.savefig('E:/UiO master merkur simulasjoner/animation/plots/mer_r' + str(usr1) + '/mer_r' + str(usr1) + '_' + str(usr3) + '_' + str(usr4) + '_' + str(i) + '.png')
        plt.close()
    else:
        plt.show()

usr1, usr2, usr3, usr4 = 1, 150, 'xz', 4 # user_input()

if usr2 == '00':
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
        path = 'E:/UiO master merkur simulasjoner/mer_r' + str(usr1) + '/tec2D/2D_' + str(usr3) + '/prop' + str(usr3) + '00' + str(i) + '.dat'
        print('Making figure ' + str(i))
        # print(path)
        if Path(path).is_file() == False:
            print('No figure ' + str(i) + ' found.')
            print('We have reached the bottom. No more files to read.')
            print('Exiting...')
            break
        else:
            get_figure(get_data(path)[usr4], name_colorbar(usr4)[0], name_colorbar(usr4)[1])
            print('Figure ' + str(i) + ' done.')
else:
    path = 'E:/UiO master merkur simulasjoner/mer_r' + str(usr1) + '/tec2D/2D_' + str(usr3) + '/prop' + str(usr3) + '00' + str(usr2) + '.dat'
    get_figure(get_data(path)[usr4], name_colorbar(usr4)[0], name_colorbar(usr4)[1])


# x_val = float(input("Choose vertical grid line:> "))
# y_val = float(input("Choose horizontal grid line:> "))


def linear_data_extraction(x_val=240, y_val=0, path='E:/UiO master merkur simulasjoner/mer_r' + str(usr1) + '/tec2D/2D_' + str(usr3) + '/prop' + str(usr3) + '00' + str(usr2) + '.dat'):
    """
    Function for extracting data from the .dat-files for a linear analysis.
    """
    data = get_data(path)[usr4]                                                     # Getting the data for the chosen parameter
    
    horisontal_data_line = data[(0 + (320 * y_val)):(319 + (320 * y_val))]          # Extracting data for horisontal line
    vertical_data_line = [data[0 + (x_val + (320 * i))] for i in range(0, 320)]     # Extracting data for vertical line

    x_axis_horisontal = get_data(path)[0]                                           # Extracting x-axis values for horisontal line
    x_axis_vertical = get_data(path)[1]                                             # Extracting y-axis values for vertical line
    x_av = np.array([x_axis_vertical[0 + (320 * i)] for i in range(0, 320)])        # x-axis for vertical line
    x_ah = np.array(x_axis_horisontal[0:319])                                       # x-axis for horisontal line

    # Plotting scheme for a horisontal line
    if y_val != 0:
        print("Data extracted at y = " + str(x_axis_vertical[y_val] / 1e3) + "km")
        plt.plot(x_ah / 1e3, horisontal_data_line)
        plt.xlabel('x / km'), plt.title(name_colorbar(usr4)[1]), plt.grid()
        plt.show()
    
    # Plotting scheme for a vertical line
    if x_val != 0:
        print("Data extracted at x = " + str(x_axis_horisontal[x_val] / 1e3) + "km")
        plt.plot(x_av / 1e3, vertical_data_line)
        plt.xlabel('y / km'), plt.title(name_colorbar(usr4)[1]), plt.grid()
        plt.show()


linear_data_extraction()







