import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

n = 1000                                  # number of time steps
merc_mass = 3.285e23                      # mass of Mercury [kg]
sat_mass = 250                            # total mass of Mio satellite [kg]
merc_rad = 2439.7 * 1e3                         # radius of Mercury [m]
G = 6.67430e-11                           # gravitational constant [m^3 kg^-1 s^-2]
mu = G * (merc_mass + sat_mass)           # gravitational parameter
h_min = 400 * 1e3                         # smallest distance from surface km
h_max = 11824 * 1e3                       # largest distance from surface km

r_1 = 400 * 1e3                                   # smallest distance from surface km
r_2 = 11824 * 1e3                                 # largest distance from surface km
semi_major = (r_2 + (merc_rad * 2) + r_1) / 2          # center of ellipse km

pos = np.zeros((n, 2))
vel = np.zeros((n, 2))
t = np.zeros(n)

# Initial conditions
pos[0, 0], pos[1, 0] = (merc_rad + h_min), 0
vel[0, 0], vel[1, 0] = 0, np.sqrt(mu * ((2 / pos[0, 0]) - (1 / semi_major)))

