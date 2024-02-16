import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("./Simulation Data/plot-y2.dat")
x = data[:, 0]*-1
y = data[:, 1]*-1

plt.scatter(x, y,s=5)
plt.title('VCOTA(FoM x Idd)')
plt.ylabel('Idd [A]')
plt.xlabel('FoM [MHz * pF / mA]')
plt.show()