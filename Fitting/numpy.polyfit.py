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
def optFun1(coefs, x):
    return coefs[0] + coefs[1] * x + coefs[2] * x**2

def optFun2(x, a,b,c):
    return a + b * x + c * x**2

def resFun1(coefs, x, y):
    return optFun1(coefs, x)  -  y

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

#plt.plot(x, y, 'or')
#plt.errorbar(x, y, yerr=sigma, fmt='ro')  # Plot red circles with constant error bars
plt.errorbar(x, y, yerr=errors, fmt='ro')  # Plot red circles with actual error bars

sigmas = np.ones(len(y))*sigma
sigmas = np.ones(len(y))

# Fit data and plot fit:
print("\nSimple fit, no errors:")
coefs = np.polyfit(x, y, 2)
print(coefs)


print("\nFit without errors, with variance/covariance matrix:")
coefs, varCov = np.polyfit(x, y, 2, cov=True, w=1/sigmas)
#print(coefs)
#print(varCov)
#print()
print("Coefficients:")
for iCoef in range(3):
    print(iCoef+1,":", coefs[iCoef], "+-", m.sqrt(varCov[iCoef][iCoef]))
    


print("\nFit without errors, with chi squared:")
coefs, resids, rank, singular_values, rcond = np.polyfit(x, y, 2, full=True)
#print(resids)
print("coefficients: ", coefs)
print("residuals: ", resids)
Chi2 = resids[0]                    # Chi^2
redChi2 = Chi2/(len(x)-rank)        # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)
#print("rank: ", rank)
#print("singular_values: ", singular_values)
#print("rcond: ", rcond)



if False:
    print("\n\nFit with constant errors:")
    coefs, resids, rank, singular_values, rcond = np.polyfit(x, y, 2, w=1/sigmas, full=True)
    print("coefficients: ", coefs)
    Chi2 = resids[0]                    # Chi^2
    redChi2 = Chi2/(len(x)-rank)        # Reduced Chi^2 = Chi^2 / (n-m)
    print("Chi2: ", Chi2)
    print("Red. Chi2: ", redChi2)
    
    coefs, varCov = np.polyfit(x, y, 2, cov=True, w=1/sigmas)
    print("\nCoefficients:")
    for iCoef in range(3):
        print(iCoef+1,":", coefs[iCoef], "+-", m.sqrt(varCov[iCoef][iCoef]))
        
        
        
    print("\n\nFit with actual errors:")
    coefs, resids, rank, singular_values, rcond = np.polyfit(x, y, 2, w=1/errors, full=True)
    print("coefficients: ", coefs)
    Chi2 = resids[0]                    # Chi^2
    redChi2 = Chi2/(len(x)-rank)        # Reduced Chi^2 = Chi^2 / (n-m)
    print("Chi2: ", Chi2)
    print("Red. Chi2: ", redChi2)
    
    coefs, varCov = np.polyfit(x, y, 2, cov=True, w=1/errors)
    print("\nCoefficients:")
    for iCoef in range(3):
        print(iCoef+1,":", coefs[iCoef], "+-", m.sqrt(varCov[iCoef][iCoef]))
    



# Plot the (last) fit:
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefs, xn)
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

dCoefs = [0,0,0]  # Not sure how to get the uncertainties in the coefficients!
print("\nCoefficients:")
for iCoef in range(3):
    print(iCoef+1,":", coefs[iCoef], "+-", dCoefs[iCoef])
    
print()
#Chi2 = sum(res.fun**2)                      # Chi^2
Chi2 = res.cost*2  # Same thing!             # Chi^2
redChi2 = Chi2/(len(x)-len(res.x))           # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2)
print("Red. Chi2: ", redChi2)

print('\nDifference in red. Chi^2: ', abs(redChi20-redChi2))


# Plot the (last) fit:
coefs = [res.x[2], res.x[1], res.x[0]]
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefs, xn)
plt.plot(xn, yn)







print("\n\nFit with scipy.optimize.leastsq():")
x0 = [-3, 0, 5]  # Initial guess for coefficients
#coefs, cov_x, infodict, mesg, ier = leastsq( resFun1, x0, args=(x, y), full_output=True )
coefs, cov_x, infodict, mesg, ier = leastsq( resFun2, x0, args=(x, y, sigmas), full_output=True )

print('Success:      ', ier)
#print('Message:      ', mesg)
print('Coefficients: ', coefs)
#print('Variance/cov: ', cov_x

#resids = resFun1(coefs, x, y)         # Residuals
resids = resFun2(coefs, x, y, sigmas)  # Residuals

Chi2    = sum(resids**2)               # Chi^2
redChi2 = Chi2/(len(x)-len(coefs))     # Reduced Chi^2 = Chi^2 / (n-m)
varCov  = cov_x * redChi2              # Variance-covariance matrix
dCoefs = np.sqrt(np.diag(varCov))      # Standard deviations on the coefficients

print("\nCoefficients:")
for iCoef in range(3):
    print(iCoef+1,":", coefs[2-iCoef], "+-", dCoefs[2-iCoef])  # Reverse order w.r.t. polyfit
    
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













print("\n\nFit with scipy.optimize.curve_fit():")
x0 = [-3, 0, 5]  # Initial guess for coefficients
coefs,varCov = curve_fit(optFun2, x, y, p0=x0, sigma=sigmas, method='lm')
#coefs, varCov, infodict, mesg, ier = curve_fit(optFun2, x, y, p0=x0, sigma=sigmas, method='lm', full_output=True)
print('coefficients: ', coefs)
print('variance/covariance: ', varCov)
dCoefs = np.sqrt(np.diag(varCov))     # Standard deviations on the coefficients

print("\nCoefficients:")
for iCoef in range(3):
    print(iCoef+1,":", coefs[2-iCoef], "+-", dCoefs[2-iCoef])  # Reverse order w.r.t. polyfit
    
print()
resids = optFun2(x, *coefs) - y  # Residuals
Chi2    = sum(resids**2)               # Chi^2
redChi2 = Chi2/(len(x)-len(coefs))     # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2, res.cost*2)
print("Red. Chi2: ", redChi2)

print('\nDifference in red. Chi^2: ', abs(redChi20-redChi2))
 

# Plot the (last) fit:
#coefs = [res.x[2], res.x[1], res.x[0]]
xn = np.linspace(0, 2, 200)
yn = np.polyval(coefs, xn)
plt.plot(xn, yn)

print()


