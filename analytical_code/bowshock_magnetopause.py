import numpy as np
import matplotlib.pyplot as plt

particle_mass = 1.67e-27    # Mass of a proton [kg]
B_0 = 3e-7                  # Mercury's magnetic field at equator [T]
rho = 3e7                 # Density of the solar wind [particle/m^3]
v_sw = 500e3                # Velocity of the solar wind [m/s]
mu_0 = 4 * np.pi * 1e-7     # Permeability in vacuum [T m/A]
R_M = 2440                  # Mercury's radius [km]


def magnetopause(B_0, rho, v_sw, mu_0, R_M):
    '''
    Function for calculating the theoretical bowshock radius and location
    for the Mercury's magnetosphere. End result is given in km.
    '''
    r_mp = np.power((2 * B_0 ** 2 * R_M ** 6) / ((rho * particle_mass) * v_sw ** 2 * mu_0), 1/6)
    return r_mp

magnetopause_merc = magnetopause(B_0, rho, v_sw, mu_0, R_M)
print(magnetopause_merc)

# Plotting the magnetopause
# r = np.linspace(1, 10, 1000)
# plt.plot(r, magnetopause_merc)
# plt.xlabel('Distance from Mercury [R$_M$]')
# plt.ylabel('Bowshock radius [R$_M$]')
# plt.title('Theoretical bowshock radius for Mercury')
# plt.show()
