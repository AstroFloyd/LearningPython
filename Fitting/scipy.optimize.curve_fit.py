#!/bin/env python3

"""Example fit using scipy.optimize.curve_fit() on a polynomial.

References:

  - https://mmas.github.io/least-squares-fitting-numpy-scipy

"""


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Function to be optimised: independent variable(s) + individual coefficients.  Needed for curve_fit.
def optFun(x, a,b,c):
    return a + b * x + c * x**2



#plt.style.use('dark_background')        # Invert colours
plt.figure(figsize=(12.5,7))             # Set png size to 1250x700; savefig has default dpi 100


# Create and plot noisy data:
trueCoefs = [-5, 1, 3]
sigma = 1.5
print("True coefficients: ", trueCoefs)
print("Sigma: ", sigma)

f = np.poly1d(trueCoefs)
xDat = np.linspace(0, 2, 20)
errors = sigma*np.random.normal(size=len(xDat))
yDat = f(xDat) + errors

#plt.errorbar(xDat, yDat, yerr=sigma, fmt='ro')  # Plot red circles with constant error bars
plt.errorbar(xDat, yDat, yerr=errors, fmt='ro')  # Plot red circles with actual error bars


# Array of measurement errors:
#sigmas = np.ones(len(yDat))        # Use constant measurement errors, == 1.
sigmas = np.ones(len(yDat))*sigma   # Use constant measurement errors, == sigma.
#sigmas = errors                  # Use measurement errors == actual error!




print("\nFit with scipy.optimize.curve_fit():")
coef0 = [-3, 0, 5]  # Initial guess for coefficients
#coefs,varCov = curve_fit(optFun, xDat, yDat, p0=coef0, sigma=sigmas, method='lm')
coefs, varCov, infodict, mesg, ier = curve_fit(optFun, xDat, yDat, p0=coef0, sigma=sigmas, method='lm', full_output=True)
print("Success: ", ier)
#print('coefficients: ', coefs)
#print('variance/covariance: ', varCov)
dCoefs = np.sqrt(np.diag(varCov))     # Standard deviations on the coefficients

resids  = (optFun(xDat, *coefs) - yDat)/sigmas   # Weighted residuals
Chi2    = sum(resids**2)                    # Chi^2
redChi2 = Chi2/(len(xDat)-len(coefs))          # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)
print("Coefficients:")
for iCoef in range(3):
    print(" ", iCoef+1,":", coefs[2-iCoef], "+-", dCoefs[2-iCoef])  # Reverse order w.r.t. polyfit




# Plot the fit:
xFit = np.linspace(0, 2, 200)
yFit = np.polyval([coefs[2],coefs[1],coefs[0]], xFit)
plt.plot(xFit, yFit)

plt.tight_layout()
#plt.show()
plt.savefig('scipy.optimize.curve_fit.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later


print()


