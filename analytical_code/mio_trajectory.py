import numpy as np
import matplotlib.pyplot as plt


def satellite_position():
    '''
    Function to calculate the position of the Mio satellite within the data grid,
    set up by the simulation. Mercury is placed in the center with (x, y) = (0, 0).
    '''
    # Prompt user to choose the rotation of the satellite orbit
    set_rot = int(input("Choose the rotation of the satellite orbit (0, 90, 180, 360): "))

    # Define the parameters
    N = 33560                                       # number of timesteps for one full orbit
    dt = 1                                          # timestep [s]
    gravitational_constant = 6.67430e-11            # gravitational constant [m^3 kg^-1 s^-2]
    m_merc = 3.285e23                               # mass of Mercury [kg]
    r_m = 2440 * 1e3                                # mercury radius [m]
    r_km = 2440                                     # mercury radius [km]
    r_1 = 400 * 1e3                                 # smallest distance from surface, periapsis [m]
    r_2 = 11824 * 1e3                               # largest distance from surface, apoapsis [m]
    semi_major = (2 * r_m + r_2 + r_1) / 2          # semi-major axis [m]
    a = r_m + r_2                                   # distance from focus to apoapsis, where the center of Mercury is the focus [m]
    timestamp_enter = 8675                          # timestamp where the satellite enters the simulation data grid (after 8675 sec)
    timestamp_exit = N - timestamp_enter            # timestamp where the satellite exits the simulation data grid (after 33560 - 8675 sec)

    # Initial satellite velocity for an elliptical orbit
    v_sat = np.sqrt(gravitational_constant * m_merc * ((2 / a) - (1 / semi_major)))

    # Satellite position and velocity arrays in 2D
    pos = np.zeros([N, 2])
    vel = np.zeros([N, 2])

    # Initial position [m] and velocity [m/s] of the satellite, chosen to be at the largest disctance from Mercury's center
    if set_rot == 0:
        # From tail, inside magnetopause
        pos[0, 0] = r_2 + r_m                              
        vel[0, 1] = v_sat
    elif set_rot == 90:
        # "Above" planet, outside bowshock to inside magnetopause
        pos[0, 1] = r_2 + r_m
        vel[0, 0] = - v_sat
    elif set_rot == 180:
        # From solar wind direction, inside nightside tail
        pos[0, 0] = - r_2 - r_m
        vel[0, 1] = - v_sat
    elif set_rot == 360:
        # "Below" planet, inside bowshock to outside magnetopause
        pos[0, 1] = - r_2 - r_m
        vel[0, 0] = v_sat

    for i in range(0, N - 1):
        # Update the acceleration, velocity and position of the satellite using the Euler-Cromer method
        r = np.linalg.norm(pos[i, :])
        acc = (-gravitational_constant * m_merc / r ** 2) * (pos[i, :] / r)
        vel[i + 1, :] = vel[i, :] + acc * dt
        pos[i + 1, :] = pos[i, :] + vel[i + 1, :] * dt

    # Extract the satellite positions within the data grid, flip the axis (with minus sign) and convert to km
    inside_datagrid = pos[timestamp_enter:timestamp_exit] / 1e3
    # inside_datagrid[:, 0] = inside_datagrid[:, 0]                     # change from clockwise to counter-clockwise rotation

    # Draw the surface of Mercury and the satellite's orbit
    mercury = plt.Circle((0, 0), r_km, color='k', fill=False, label="Mercury's surface", zorder=4)
    plt.gcf().gca().add_patch(mercury)
    plt.plot(inside_datagrid[:, 0], inside_datagrid[:, 1], label="Mio's orbit"), plt.axis('equal')
    plt.plot(inside_datagrid[0 * 1800, 0], inside_datagrid[0 * 1800, 1], 'o', color='limegreen', markersize=3)
    for i in range(1, 10):
        # Mark every 30 minutes with a red dot
        plt.plot(inside_datagrid[i * 1800, 0], inside_datagrid[i * 1800, 1], 'ro', markersize=3)
    
    plt.xticks(np.array([-2 * r_km, -1 * r_km, 0, 1 * r_km, 2 * r_km, 3 * r_km, 4 * r_km, 5 * r_km]), [-2, -1, 0, 1, 2, 3, 4, 5])  # Setting up the x-axis ticks
    plt.yticks(np.array([-2 * r_km, -1 * r_km, 0, 1 * r_km, 2 * r_km, 3 * r_km]), [-2, -1, 0, 1, 2, 3])  # Setting up the y-axis ticks
    plt.title("Time evolution of Mio's orbit around Mercury"), plt.xlabel('$R_M$'), plt.ylabel('$R_M$'), plt.grid(color='0.90', zorder=1)
    plt.savefig('mio_trajectory.pdf')
    plt.show()


satellite_position()



'''
    def elliptical():
'''
    # Simple function for plotting the Mio trajectory in an elliptical orbit around Mercury.
'''
    # Define the parameters
    r_m = 2440          # mercury radius km
    r_1 = 400           # smallest distance from surface km
    r_2 = 11824         # largest distance from surface km
    r_a = r_2 + r_m     # distance from focus to aphelion
    r_p = r_1 + r_m     # distance from focus to perihelion
    semi_major = (r_2 + (r_m * 2) + r_1) / 2    # center of ellipse km

    # Eccentricity
    e = (r_a / semi_major) - 1
    print(e)

    # semi-minor axis [km]
    semi_minor = semi_major * np.sqrt(1 - e**2)
    print(semi_minor, semi_major)

    # Foci
    f_1 = np.sqrt(semi_major ** 2 - semi_minor ** 2)
    f_2 = - f_1

    # Plotting scheme
    deg = np.linspace(0, 2 * np.pi, 1000)
    x = semi_major * np.cos(deg)
    y = semi_minor * np.sin(deg)
    mercury = plt.Circle((f_2, 0), r_m, color='k', fill=False)
    
    plt.plot(x, y, alpha=0.4)
    plt.scatter(x, y, s=1, color='r')
    plt.plot(f_1, 0, 'ro'), plt.plot(f_2, 0, 'ro')
    plt.gcf().gca().add_patch(mercury)
    plt.grid(), plt.axis('equal')
    plt.show()


    def satellite_orbital():
'''
    # Function for calculating the velocity of the Mio satellite in its orbit around Mercury.
    # Starting at the smallest distance from the surface (highest velocity).
'''
    # Define the parameters
    r_m = 2440 * 1e3                                  # mercury radius km
    r_1 = 400 * 1e3                                   # smallest distance from surface km
    r_2 = 11824 * 1e3                                 # largest distance from surface km
    semi_major = (r_2 + (r_m * 2) + r_1) / 2          # center of ellipse km
    m_merc = 3.285e23                                 # mass of Mercury [kg]
    m_sat = 250                                       # total mass of Mio satellite [kg]
    G = 6.67430e-11                                   # gravitational constant [m^3 kg^-1 s^-2]
    e = 0.6679139382600561                            # eccentricity of Mio's orbit
    deg = np.linspace(0, 2 * np.pi, 10000)             # degrees for plotting
    mu = G * (m_merc + m_sat)                         # gravitational parameter
    
    r_sat = ((semi_major * (1 - e ** 2)) / (1 + e * np.cos(deg))) / 1e3       # distance from Mercury's center to satellite in km
    v_sat = np.sqrt(mu * ((2 / r_sat) - (1 / semi_major))) / 1e3              # velocity of satellite in km/s
    x = r_sat * np.cos(deg)                                                   # x-coordinates
    y = r_sat * np.sin(deg)                                                   # y-coordinates

    # semi-minor axis [km]
    semi_minor = semi_major * np.sqrt(1 - e**2)

    f_1 = np.sqrt(semi_major ** 2 - semi_minor ** 2)
    f_2 = - f_1
    
    # mercury = plt.Circle((0, 0), r_m, color='k', fill=False, label='Mercury surface [km]')
    # plt.gcf().gca().add_patch(mercury)
    # plt.plot(deg, v_sat / 1e3), plt.grid(), plt.show()
    # plt.vlines(r_m / 1e3, min(v_sat), max(v_sat), color = 'r', linestyle = '--', label='Mercury surface [km]')
    # plt.xlabel('Distance from Mercury\'s center [km]'), plt.ylabel('Velocity [km/s]')
    # plt.plot(r_sat, v_sat, label='Satellite velocity'), plt.legend(), plt.grid(), plt.show()
    plt.scatter(r_sat, v_sat, s=1, color='r'), plt.legend(), plt.grid(), plt.show()
    # plt.scatter(x, y, s=1, color='r', label='Mio trajectory'), plt.legend(), plt.grid(), plt.axis('equal'), plt.show()

    # arc length (not dependent on velocity!!)
    # theta = theta_2 - theta_1
    # s = r_sat * theta

    
# elliptical()
# satellite_orbital()
# satellite_sampling()
'''
