#!/bin/env python3

"""Example fit using a scipy.optimize.curve_fit() on a multivariate function.

"""


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Function to be optimised: independent variable(s) + individual coefficients.  Needed for curve_fit.
def optFun(x, a,b,c):
    return a * np.sin(b*x[0]) * x[1] + c
    #return a + b * x + c * x**2




# plt.style.use('dark_background')        # Invert colours
plt.figure(figsize=(12.5,7))             # Set png size to 1250x700; savefig has default dpi 100


# Create and plot noisy data:
nDat  = 100
tDat  = np.linspace(0, 12, nDat) + 6  # Time
xDat1 = np.sin((tDat-6)/12*np.pi) * np.pi / 4  # for 6-18, 0-pi/4 = 0-45°
xDat2 = np.random.uniform(0, 1, nDat)

# print(tDat)
# print(xDat1*57)
# print(xDat2)

trueCoefs = [800, 1, 0]
sigma = 100
print("True coefficients: ", trueCoefs)
print("Sigma: ", sigma)

errors = sigma*np.random.normal(size=nDat)
yDat = trueCoefs[0] * np.sin(trueCoefs[1] * xDat1) * xDat2 + trueCoefs[2] + errors

yDat[yDat < 0] = 0

# print(yDat)

xDat = np.array([xDat1,xDat2])

# print()
# print(xDat)

# plt.errorbar(tDat, yDat, yerr=sigma, fmt='ro')  # Plot red circles with constant error bars
plt.errorbar(tDat, yDat, yerr=errors, fmt='ro')  # Plot red circles with actual error bars

plt.plot(tDat, xDat1*57)
plt.plot(tDat, xDat2*100)

# Array of measurement errors:
#sigmas = np.ones(len(yDat))        # Use constant measurement errors, == 1.
sigmas = np.ones(len(yDat))*sigma   # Use constant measurement errors, == sigma.
#sigmas = errors                  # Use measurement errors == actual error!




print("\nFit with scipy.optimize.curve_fit():")
coef0 = [500, 2, -1]  # Initial guess for coefficients
coefs, varCov, infodict, mesg, ier = curve_fit(optFun, xDat, yDat, p0=coef0, sigma=sigmas, method='lm', full_output=True)
print("Success: ", ier)
#print("Success: ", mesg)
print('coefficients: ', coefs)
#print('variance/covariance: ', varCov)
#print("Infodict: ", infodict.keys(), infodict)

dCoefs = np.sqrt(np.diag(varCov))     # Standard deviations on the coefficients

resids  = (optFun(xDat, *coefs) - yDat)/sigmas    # Weighted residuals
Chi2    = sum(resids**2)                          # Chi^2
redChi2 = Chi2/(len(yDat)-len(coefs))             # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)
print("Coefficients:")
for iCoef in range(3):
    print(" ", iCoef+1,":", coefs[iCoef], "+-", dCoefs[iCoef])  # Reverse order w.r.t. polyfit




# Plot the fit:
yFit = optFun(xDat, *coefs)
plt.plot(tDat, yFit)

plt.tight_layout()
#plt.show()
plt.savefig('scipy.optimize.curve_fit-multivar.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later

#print(xFit)
#print(yFit)

print()


