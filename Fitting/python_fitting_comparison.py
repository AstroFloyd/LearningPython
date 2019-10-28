#!/bin/env python3

"""Example fits using a polynomial and numpy.polyfit as well as scipy.optimize.[curve_fit,leastsq,least_squares]

References:

  - https://mmas.github.io/least-squares-fitting-numpy-scipy

"""


import numpy as np
from scipy.optimize import least_squares, curve_fit, leastsq
import matplotlib.pyplot as plt
import math as m


# Function to be optimised: coefficients as a vector + indipendent variable(s).  Needed for leastsq() and
# least_squares().
def optFun1(coefs, x):
    return coefs[0] + coefs[1] * x + coefs[2] * x**2

# Function to be optimised: independent variable(s) + individual coefficients.  Needed for curve_fit.
def optFun2(x, a,b,c):
    return a + b * x + c * x**2

# Function to compute residuals, for leastsq() and least_squares(), without sigma.
def resFun1(coefs, x, y):
    return optFun1(coefs, x)  -  y

# Function to compute residuals, for leastsq() and least_squares(), with sigma.
def resFun2(coefs, x, y, sigmas):
    return (optFun1(coefs, x)  -  y)/sigmas



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




print("\n\nFit with numpy.polyfit():")
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

redChi20 = redChi2  # Store for comparison to other methods





print("\n\nFit with scipy.optimize.curve_fit():")
x0 = [-3, 0, 5]  # Initial guess for coefficients
#coefs,varCov = curve_fit(optFun2, x, y, p0=x0, sigma=sigmas, method='lm')
coefs, varCov, infodict, mesg, ier = curve_fit(optFun2, x, y, p0=x0, sigma=sigmas, method='lm', full_output=True)
print("Success: ", ier)
#print('coefficients: ', coefs)
#print('variance/covariance: ', varCov)
dCoefs = np.sqrt(np.diag(varCov))     # Standard deviations on the coefficients

resids  = (optFun2(x, *coefs) - y)/sigmas   # Weighted residuals
Chi2    = sum(resids**2)                    # Chi^2
redChi2 = Chi2/(len(x)-len(coefs))          # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)
print("Coefficients:")
for iCoef in range(3):
    print(" ", iCoef+1,":", coefs[2-iCoef], "+-", dCoefs[2-iCoef])  # Reverse order w.r.t. polyfit
    
print('\nDifference in red. Chi^2: ', abs(redChi20-redChi2))
 

# Plot the (last) fit:
#coefs = [res.x[2], res.x[1], res.x[0]]
xn = np.linspace(0, 2, 200)
yn = np.polyval([coefs[2],coefs[1],coefs[0]], xn)
plt.plot(xn, yn)









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
dCoefs = np.sqrt(np.diag(varCov))      # Standard deviations on the coefficients

print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)

print("Coefficients:")
for iCoef in range(3):
    print(" ", iCoef+1,":", coefs[2-iCoef], "+-", dCoefs[2-iCoef])  # Reverse order w.r.t. polyfit
    
print('\nDifference in red. Chi^2: ', abs(redChi20-redChi2))


# Plot the (last) fit:
coefs = [coefs[2], coefs[1], coefs[0]]
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefs, xn)
plt.plot(xn, yn)



















print("\n\nFit with scipy.optimize.least_squares():")
x0 = [-3, 0, 5]  # Initial guess for coefficients
#res = least_squares(resFun1, x0, args=(x, y), method='lm')
res = least_squares(resFun2, x0, args=(x, y, sigmas), method='lm')
#print('res: ', res)

print('Success:      ', res.success)
#print('Cost:         ', res.cost)  # = Chi^2/2 !
#print('Optimality:   ', res.optimality)
#print('Coefficients: ', res.x)
#print('Grad:         ', res.grad)
#print('Residuals:    ', res.fun)

#Chi2 = sum(res.fun**2)                      # Chi^2
Chi2 = res.cost*2  # Same thing!             # Chi^2
redChi2 = Chi2/(len(x)-len(res.x))           # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)

dCoefs = [0,0,0]  # Not sure how to get the uncertainties in the coefficients!
print("Coefficients:")
for iCoef in range(3):
    print(" ", iCoef+1,":", coefs[iCoef], "+-", dCoefs[iCoef])
    

print('\nDifference in red. Chi^2: ', abs(redChi20-redChi2))


# Plot the (last) fit:
coefs = [res.x[2], res.x[1], res.x[0]]
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefs, xn)
plt.plot(xn, yn)







plt.tight_layout()
#plt.show()
plt.savefig('python_fitting_comaprison.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later


print()


