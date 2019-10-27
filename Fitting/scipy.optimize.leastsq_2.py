#!/bin/env python3

# https://stackoverflow.com/a/28242456/1386750

from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt


# Function for quadratic fit.  tpl is a tuple that contains the parameters of the fit.
def funcQuad(tpl,x):
    return tpl[0]*x**2 + tpl[1]*x + tpl[2]

# ErrorFunc is the diference between the function and the y data.
def ErrorFunc(tpl,x,y):
    return funcQuad(tpl, x) - y



# Create and plot data:
x = np.array([1.0, 2.5, 3.5, 4.0, 1.1, 1.8, 2.2, 3.7])
y = np.array([6.008, 15.722, 27.130, 33.772, 5.257, 9.549, 11.098, 28.828])
plt.plot(x, y, 'bo')


# Fit the data to the function funcQuad():
tplInitial = (1.0, 2.0, 3.0)  # Initial guess
tplFinal,success = leastsq(ErrorFunc, tplInitial[:], args=(x,y))  # Make the fit
print("quadratic fit" , tplFinal, success)

# Plot fit:
xx = np.linspace(x.min(), x.max(), 50)
yy = funcQuad(tplFinal, xx)
plt.plot(xx, yy, 'g-')

plt.tight_layout()
# plt.show()
plt.savefig('scipy.optimize.leastsq_2.png')     # Save the plot as png
plt.close()                                     # Close the plot in order to start a new one later


print()


