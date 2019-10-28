#!/bin/env python3

"""Example fit using a polynomial in Numpy: numpy.polyfit

References:

  - https://mmas.github.io/least-squares-fitting-numpy-scipy

"""


import numpy as np
from scipy.optimize import least_squares, curve_fit, leastsq
import matplotlib.pyplot as plt
import math as m


# Function for computing residuals:
def optFun(c, x):
    return c[0] + c[1] * x + c[2] * x**2

def resFun1(c, x, y):
    return optFun(c, x)  -  y

def resFun2(c, x, y, sigmas):
    return (optFun(c, x)  -  y)/sigmas


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

sigmas = np.ones(len(y))*sigma
sigmas = np.ones(len(y))

# Fit data and plot fit:
print("\nSimple fit, no errors:")
coefficients = np.polyfit(x, y, 2)
print(coefficients)


print("\nFit without errors, with variance/covariance matrix:")
coefficients, varCov = np.polyfit(x, y, 2, cov=True, w=1/sigmas)
print(coefficients)
print(varCov)
print()
print("Coefficients:")
for iCoef in range(3):
    print(iCoef+1,":", coefficients[iCoef], "+-", m.sqrt(varCov[iCoef][iCoef]))
    


print("\nFit without errors, with chi squared:")
coefficients, residuals, rank, singular_values, rcond = np.polyfit(x, y, 2, full=True)
#print(residuals)
print("coefficients: ", coefficients)
print("residuals: ", residuals)
Chi2 = residuals[0]                    # Chi^2
redChi2 = Chi2/(len(x)-rank)           # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)
print("rank: ", rank)
print("singular_values: ", singular_values)
print("rcond: ", rcond)



if False:
    print("\n\nFit with constant errors:")
    coefficients, residuals, rank, singular_values, rcond = np.polyfit(x, y, 2, w=1/sigmas, full=True)
    print("coefficients: ", coefficients)
    Chi2 = residuals[0]                    # Chi^2
    redChi2 = Chi2/(len(x)-rank)           # Reduced Chi^2 = Chi^2 / (n-m)
    print("Chi2: ", Chi2)
    print("Red. Chi2: ", redChi2)
    
    coefficients, varCov = np.polyfit(x, y, 2, cov=True, w=1/sigmas)
    print("\nCoefficients:")
    for iCoef in range(3):
        print(iCoef+1,":", coefficients[iCoef], "+-", m.sqrt(varCov[iCoef][iCoef]))
        
        
        
    print("\n\nFit with actual errors:")
    coefficients, residuals, rank, singular_values, rcond = np.polyfit(x, y, 2, w=1/errors, full=True)
    print("coefficients: ", coefficients)
    Chi2 = residuals[0]                    # Chi^2
    redChi2 = Chi2/(len(x)-rank)           # Reduced Chi^2 = Chi^2 / (n-m)
    print("Chi2: ", Chi2)
    print("Red. Chi2: ", redChi2)
    
    coefficients, varCov = np.polyfit(x, y, 2, cov=True, w=1/errors)
    print("\nCoefficients:")
    for iCoef in range(3):
        print(iCoef+1,":", coefficients[iCoef], "+-", m.sqrt(varCov[iCoef][iCoef]))
    



# Plot the (last) fit:
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefficients, xn)
plt.plot(xn, yn)

redChi20 = redChi2



print("\n\nFit with scipy.optimize.least_squares():")
# Compute a standard least-squares solution using scipy.optimize:
x0 = [-3, 0, 5]  # Initial guess for coefficients
#res = least_squares(resFun1, x0, args=(x, y), method='lm')
res = least_squares(resFun2, x0, args=(x, y, sigmas), method='lm')
#print('res: ', res)

print('Success:      ', res.success)
print('Cost:         ', res.cost)
#print('Optimality:   ', res.optimality)
print('Coefficients: ', res.x)
#print('Grad:         ', res.grad)
#print('Residuals:    ', res.fun)

Chi2 = sum(res.fun**2)
redChi2 = Chi2/(len(x)-len(res.x))           # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2, res.cost*2)
print("Red. Chi2: ", redChi2)

print('\nDifference in red. Chi^2: ', abs(redChi20-redChi2))


# Plot the (last) fit:
coefficients = [res.x[2], res.x[1], res.x[0]]
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefficients, xn)
plt.plot(xn, yn)







print("\n\nFit with scipy.optimize.leastsq():")
x0 = [-3, 0, 5]  # Initial guess for coefficients
#coefs, cov_x, infodict, mesg, ier = leastsq( resFun1, x0, args=(x, y), full_output=True )
coefs, cov_x, infodict, mesg, ier = leastsq( resFun2, x0, args=(x, y, sigmas), full_output=True )

print('Success:      ', ier)
#print('Message:      ', mesg)
print('Coefficients: ', coefs)
#print('Variance/cov: ', cov_x

#residuals = resFun1(coefs, x, y)
residuals = resFun2(coefs, x, y, sigmas)

Chi2    = sum(residuals**2)            # Chi^2
redChi2 = Chi2/(len(x)-len(coefs))     # Reduced Chi^2 = Chi^2 / (n-m)
varCov  = cov_x * redChi2              # Variance-covariance matrix
dCoeffs = np.sqrt(np.diag(varCov))     # Standard deviations on the coefficients

print("\nCoefficients:")
for iCoef in range(3):
    print(iCoef+1,":", coefs[2-iCoef], "+-", dCoeffs[2-iCoef])  # Reverse order w.r.t. polyfit
    
print()
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)

print('\nDifference in red. Chi^2: ', abs(redChi20-redChi2))


# Plot the (last) fit:
coefs = [res.x[2], res.x[1], res.x[0]]
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefs, xn)
plt.plot(xn, yn)








plt.tight_layout()
#plt.show()
plt.savefig('numpy.polyfit.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later


print()


