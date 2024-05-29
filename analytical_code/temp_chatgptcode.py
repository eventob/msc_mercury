import matplotlib.pyplot as plt
import numpy as np

# Example data
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-2 * (X**2 + Y**2))

# Creating the contourf plot without white lines
plt.figure()
contourf = plt.contourf(X, Y, Z, levels=15, antialiased=True)
plt.colorbar(contourf)
plt.title('Contour Plot without White Lines')
plt.show()
