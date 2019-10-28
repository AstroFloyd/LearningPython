#!/bin/env python3

"""Example fit using a polynomial in Numpy: numpy.polyfit

References:

  - https://mmas.github.io/least-squares-fitting-numpy-scipy

"""


import numpy as np
import matplotlib.pyplot as plt
import math as m


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

#plt.plot(x, y, 'or')
#plt.errorbar(x, y, yerr=sigma, fmt='ro')  # Plot red circles with constant error bars
plt.errorbar(x, y, yerr=errors, fmt='ro')  # Plot red circles with actual error bars

#sigmas = np.ones(len(y))        # Use constant measurement errors, == 1.
sigmas = np.ones(len(y))*sigma   # Use constant measurement errors, == sigma.
#sigmas = errors                  # Use measurement errors == actual error!


# Fit data and plot fit:
print("\nSimple fit, no errors:")
coefs = np.polyfit(x, y, 2)
print(coefs)


print("\nFit with polyfit(), without errors, with variance/covariance matrix:")
coefs, varCov = np.polyfit(x, y, 2, cov=True, w=1/sigmas)
#print(coefs)
#print(varCov)
#print()
print("Coefficients:")
for iCoef in range(3):
    print(" ", iCoef+1,":", coefs[iCoef], "+-", m.sqrt(varCov[iCoef][iCoef]))
    


print("\nFit with polyfit(), without errors, with chi squared:")
coefs, resids, rank, singular_values, rcond = np.polyfit(x, y, 2, full=True)
#print(resids)
print("coefficients: ", coefs)
#print("residuals: ", resids)
Chi2 = resids[0]                    # Chi^2
redChi2 = Chi2/(len(x)-rank)        # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)
#print("rank: ", rank)
#print("singular_values: ", singular_values)
#print("rcond: ", rcond)




print("\n\nFit with polyfit(), with errors:")
coefs, resids, rank, singular_values, rcond = np.polyfit(x, y, 2, w=1/sigmas, full=True)
#print("coefficients: ", coefs)
Chi2 = resids[0]                    # Chi^2
redChi2 = Chi2/(len(x)-rank)        # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)

coefs, varCov = np.polyfit(x, y, 2, cov=True, w=1/sigmas)
print("Coefficients:")
for iCoef in range(3):
    print(" ", iCoef+1,":", coefs[iCoef], "+-", m.sqrt(varCov[iCoef][iCoef]))
        

# Plot the (last) fit:
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefs, xn)
plt.plot(xn, yn)




plt.tight_layout()
#plt.show()
plt.savefig('numpy.polyfit.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later


print()


