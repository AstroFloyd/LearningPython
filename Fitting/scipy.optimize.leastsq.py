#!/bin/env python3

"""Example fit using a polynomial and scipy.optimize.leastsq()

References:

  - https://mmas.github.io/least-squares-fitting-numpy-scipy

"""


import numpy as np
from scipy.optimize import least_squares, curve_fit, leastsq
import matplotlib.pyplot as plt
import math as m


# Function to be optimised: coefficients as a vector + indipendent variable(s).  Needed for leastsq() and
# least_squares().
def optFun(coefs, x):
    return coefs[0] + coefs[1] * x + coefs[2] * x**2

# Function to compute residuals, for leastsq() and least_squares(), without sigma.
def resFun1(coefs, x, y):
    return optFun(coefs, x)  -  y

# Function to compute residuals, for leastsq() and least_squares(), with sigma.
def resFun2(coefs, x, y, sigmas):
    return (optFun(coefs, x)  -  y)/sigmas



#plt.style.use('dark_background')        # Invert colours
plt.figure(figsize=(12.5,7))             # Set png size to 1250x700; savefig has default dpi 100


# Create and plot noisy data:
trueCoefs = [-5, 1, 3]
sigma = 1.5
print("True coefficients: ", trueCoefs)
print("Sigma: ", sigma)

f = np.poly1d(trueCoefs)
x = np.linspace(0, 2, 20)
errors = sigma*np.random.normal(size=len(x))
y = f(x) + errors

#plt.errorbar(x, y, yerr=sigma, fmt='ro')  # Plot red circles with constant error bars
plt.errorbar(x, y, yerr=errors, fmt='ro')  # Plot red circles with actual error bars


# Array of measurement errors:
#sigmas = np.ones(len(y))        # Use constant measurement errors, == 1.
sigmas = np.ones(len(y))*sigma   # Use constant measurement errors, == sigma.
#sigmas = errors                  # Use measurement errors == actual error!




print("\n\nFit with scipy.optimize.leastsq():")
x0 = [-3, 0, 5]  # Initial guess for coefficients
#coefs, cov_x, infodict, mesg, ier = leastsq( resFun1, x0, args=(x, y), full_output=True )
coefs, cov_x, infodict, mesg, ier = leastsq( resFun2, x0, args=(x, y, sigmas), full_output=True )

print('Success:      ', ier)
#print('Message:      ', mesg)
#print('Coefficients: ', coefs)
#print('Variance/cov: ', cov_x

#resids = resFun1(coefs, x, y)         # Residuals
resids = resFun2(coefs, x, y, sigmas)  # Residuals

Chi2    = sum(resids**2)               # Chi^2
redChi2 = Chi2/(len(x)-len(coefs))     # Reduced Chi^2 = Chi^2 / (n-m)
varCov  = cov_x * redChi2              # Variance-covariance matrix
dCoefs  = np.sqrt(np.diag(varCov))     # Standard deviations on the coefficients

print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)

print("Coefficients:")
for iCoef in range(3):
    print(" ", iCoef+1,":", coefs[2-iCoef], "+-", dCoefs[2-iCoef])  # Reverse order w.r.t. polyfit
    

# Plot the fit:
coefs = [coefs[2], coefs[1], coefs[0]]
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefs, xn)
plt.plot(xn, yn)




plt.tight_layout()
#plt.show()
plt.savefig('scipy.optimize.leastsq.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later


print()


