#!/bin/env python3

"""Example fit using scipy.optimize.leastsq()

References:

  - https://mmas.github.io/least-squares-fitting-numpy-scipy

"""


import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import math as m


def optFun(x, a, b, c):
    return a*x**2 + b*x + c

def residual(p, x, y):
    return y - optFun(x, *p)


# Create and plot noisy data:
trueCoefs = [-5, 1, 3]
sigma = 1.5
print("True coefficients: ", trueCoefs)
print("Sigma: ", sigma)

f = np.poly1d(trueCoefs)
x = np.linspace(0, 2, 20)
errors = sigma*np.random.normal(size=len(x))
y = f(x) + errors

#plt.plot(x, y, 'or')
#plt.errorbar(x, y, yerr=sigma, fmt='ro')  # Plot red circles with constant error bars
plt.errorbar(x, y, yerr=errors, fmt='ro')  # Plot red circles with actual error bars

sigmas = np.ones(len(y))*sigma



# plt.style.use('dark_background')        # Invert colours

p0 = [1., 1., 1.]
popt, success = optimize.leastsq( residual, p0, args=(x, y) )
print(popt)
print(success)

# [-5.98229569  3.14299536  2.16551107]

xn = x
yn = optFun(xn, *popt)

plt.plot(xn, yn)


plt.tight_layout()
#plt.show()
plt.savefig('scipy.optimize.leastsq.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later


print()

