#!/bin/env python3

"""Example fit using a scipy.optimize.curve_fit() on a multivariate function.

"""


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Function to be optimised: independent variable(s) + individual coefficients.  Needed for curve_fit.
def optFun(x, a,b,c):
    return a + b * x + c * x**2



# plt.style.use('dark_background')        # Invert colours
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

# plt.errorbar(x, y, yerr=sigma, fmt='ro')  # Plot red circles with constant error bars
plt.errorbar(x, y, yerr=errors, fmt='ro')  # Plot red circles with actual error bars


# Array of measurement errors:
#sigmas = np.ones(len(y))        # Use constant measurement errors, == 1.
sigmas = np.ones(len(y))*sigma   # Use constant measurement errors, == sigma.
#sigmas = errors                  # Use measurement errors == actual error!




print("\nFit with scipy.optimize.curve_fit():")
x0 = [-3, 0, 5]  # Initial guess for coefficients
#coefs,varCov = curve_fit(optFun, x, y, p0=x0, sigma=sigmas, method='lm')
coefs, varCov, infodict, mesg, ier = curve_fit(optFun, x, y, p0=x0, sigma=sigmas, method='lm', full_output=True)
print("Success: ", ier)
#print('coefficients: ', coefs)
#print('variance/covariance: ', varCov)
dCoefs = np.sqrt(np.diag(varCov))     # Standard deviations on the coefficients

resids  = (optFun(x, *coefs) - y)/sigmas    # Weighted residuals
Chi2    = sum(resids**2)                    # Chi^2
redChi2 = Chi2/(len(x)-len(coefs))          # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)
print("Coefficients:")
for iCoef in range(3):
    print(" ", iCoef+1,":", coefs[2-iCoef], "+-", dCoefs[2-iCoef])  # Reverse order w.r.t. polyfit




# Plot the fit:
xn = np.linspace(0, 2, 200)
yn = np.polyval([coefs[2],coefs[1],coefs[0]], xn)
plt.plot(xn, yn)

plt.tight_layout()
#plt.show()
plt.savefig('scipy.optimize.curve_fit-multivar.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later


print()


