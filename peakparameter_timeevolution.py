import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.optimize as opt
import scipy.interpolate as interp

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
        data *= (Nc / 1e6)                  # From non-dim to million/m^3
    elif 3 < index < 16:
        data *= jc                  # From non-dim to A/m^2
    elif 15 < index < 19:
        data *= Ec                  # From non-dim to V/m
    elif 18 < index < 28:
        data *= (Bc * 1e9)          # From non-dim to nT
    
    return x, y, data


def peak_over_time(data, x_data, set_yval=159, set_xval=159):
    """
    Function for extracting the position of the max value of the data at a given time frame.
    """
    j = 0
    X = ((x_data - (239.625*xc))) / 1e3
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
    s1_bowshock = np.zeros(len(timesteps))
    s6_bowshock = np.zeros(len(timesteps))
    s5_bowshock = np.zeros(len(timesteps))
    s8_bowshock = np.zeros(len(timesteps))
    s9_bowshock = np.zeros(len(timesteps))
    t = np.linspace(1, len(timesteps), len(timesteps))
    for i in timesteps:
        print('Getting max values for ' + str(i), 'j = ' + str(j))
        if j <= 172:
            s1 = get_data('E:/UiO master merkur simulasjoner/mer_r1' + '/tec2D/2D_xy' + '/propxy'  + '00' + str(i) + '.dat', 2)[2]
            h1_line = s1[(0 + (320 * set_yval)):(320 + (320 * set_yval))]                       # Fetch the line of data at the chosen y-value
            h1_peak = np.max(h1_line[0:159])                                                    # Find the max value of the data for that time frame
            bowshock_max_1 = np.where(s1 == h1_peak)                                            # Find the position of the max value
            s1_bowshock[j] = X[bowshock_max_1][0] / md
        if j <= 110:
            s6 = get_data('E:/UiO master merkur simulasjoner/mer_r6' + '/tec2D/2D_xy' + '/propxy'  + '00' + str(i) + '.dat', 2)[2]
            h6_line = s6[(0 + (320 * set_yval)):(320 + (320 * set_yval))]                       # Fetch the line of data at the chosen y-value
            h6_peak = np.max(h6_line[0:159])                                                    # Find the max value of the data for that time frame
            bowshock_max_6 = np.where(s6 == h6_peak)                                            # Find the position of the max value
            s6_bowshock[j] = X[bowshock_max_6][0] / md
        if j <= 150:
            s5 = get_data('E:/UiO master merkur simulasjoner/mer_r5' + '/tec2D/2D_xy' + '/propxy'  + '00' + str(i) + '.dat', 2)[2]
            h5_line = s5[(0 + (320 * set_yval)):(320 + (320 * set_yval))]                       # Fetch the line of data at the chosen y-value
            h5_peak = np.max(h5_line[0:159])                                                    # Find the max value of the data for that time frame
            bowshock_max_5 = np.where(s5 == h5_peak)                                            # Find the position of the max value
            s5_bowshock[j] = X[bowshock_max_5][0] / md
        if j <= 151:
            s8 = get_data('E:/UiO master merkur simulasjoner/mer_r8' + '/tec2D/2D_xy' + '/propxy'  + '00' + str(i) + '.dat', 2)[2]
            h8_line = s8[(0 + (320 * set_yval)):(320 + (320 * set_yval))]                       # Fetch the line of data at the chosen y-value
            h8_peak = np.max(h8_line[0:159])                                                    # Find the max value of the data for that time frame
            bowshock_max_8 = np.where(s8 == h8_peak)                                            # Find the position of the max value
            s8_bowshock[j] = X[bowshock_max_8][0] / md
        if j <= 154:
            s9 = get_data('E:/UiO master merkur simulasjoner/mer_r9' + '/tec2D/2D_xy' + '/propxy'  + '00' + str(i) + '.dat', 2)[2]
            h9_line = s9[(0 + (320 * set_yval)):(320 + (320 * set_yval))]                       # Fetch the line of data at the chosen y-value
            h9_peak = np.max(h9_line[0:159])                                                    # Find the max value of the data for that time frame
            bowshock_max_9 = np.where(s9 == h9_peak)                                            # Find the position of the max value
            s9_bowshock[j] = X[bowshock_max_9][0] / md
        print('Done, advancing j...')
        if j == 172:
            break
        j += 1
    
    return t, s1_bowshock, s6_bowshock, s5_bowshock, s8_bowshock, s9_bowshock



# all_zero = np.where(h_val == 0)
# h_val = np.delete(h_val, all_zero)
# t = np.delete(t, all_zero)


# Polynomial fit
# p = np.polyfit(t, h_val, 3)
# f = np.poly1d(p)
# print(p, f)

def poly_func(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def exp_func(x, a, b, c):
    return a * np.exp(b * -x) + c

def gauss_func(x, a, b, c):
    return a * np.exp(-b * (x - c)**2)

def log_func(x, a, b, c):
    return a * np.log(b * x) + c

# popt, pcov = opt.curve_fit(log_func, t, h_val) # , p0=[np.max(h_val), np.mean(t), np.std(t)])
# cs = interp.CubicSpline(t, h_val)
# print(popt)

# Plotting
# plt.plot(t, cs(t), label='Interpolated data', color='blue', linewidth=1)
# plt.plot(t, log_func(t, *popt), label='Fitted data', color='red', linewidth=1)
path = 'E:/UiO master merkur simulasjoner/mer_r1' + '/tec2D/2D_xy' + '/propxy'  + '00' + '001' + '.dat'
x_data = get_data(path, 2)[0]
t, s1_bowshock, s6_bowshock, s5_bowshock, s8_bowshock, s9_bowshock = peak_over_time(get_data(path, 2)[2], x_data)

# Remove all excessive values
all_1zero = np.where(s1_bowshock == 0)
run_1away = np.where(s1_bowshock >= -1)
s1_bowshock[all_1zero] = np.nan
s1_bowshock[run_1away] = np.nan

all_6zero = np.where(s6_bowshock == 0)
run_6away = np.where(s6_bowshock >= -1)
s6_bowshock[all_6zero] = np.nan
s6_bowshock[run_6away] = np.nan

all_5zero = np.where(s5_bowshock == 0)
run_5away = np.where(s5_bowshock >= -1)
s5_bowshock[all_5zero] = np.nan
s5_bowshock[run_5away] = np.nan

all_8zero = np.where(s8_bowshock == 0)
run_8away = np.where(s8_bowshock >= -1)
s8_bowshock[all_8zero] = np.nan
s8_bowshock[run_8away] = np.nan

all_9zero = np.where(s9_bowshock == 0)
run_9away = np.where(s9_bowshock >= -1)
s9_bowshock[all_9zero] = np.nan
s9_bowshock[run_9away] = np.nan


plt.grid()
plt.scatter(t, -s1_bowshock, label='S1', s=4, alpha=0.6, color='blue')
plt.scatter(t, -s6_bowshock, label='S6', s=4, alpha=0.6, color='red')
plt.scatter(t, -s5_bowshock, label='S5', s=4, alpha=0.6, color='green')
plt.scatter(t, -s8_bowshock, label='S8', s=4, alpha=0.6, color='orange')
plt.scatter(t, -s9_bowshock, label='S9', s=4, alpha=0.6, color='purple')
plt.title('Estimated bowshock distance over time frames')
plt.legend(loc='best', fontsize='small'), plt.xlabel('Time frame [$\\omega_{c, i}$t]'), plt.ylabel('Mercury radius $R_M$')
plt.savefig('estimated_bowshock_position.pdf')
plt.show()
