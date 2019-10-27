#!/bin/env python3

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

"""Solve a curve fitting problem using robust loss function to take care of outliers in the data. Define the
model function as y = a + b * exp(c * t), where t is a predictor variable, y is an observation and a, b, c are
parameters to estimate.
"""

import numpy as np
from scipy.optimize import least_squares


# Function which generates the data with noise and outliers:
def gen_data(x, a, b, c, noise=0, n_outliers=0, random_state=0):
    y = a + b*x + c*x**2
    
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(x.size)
    outliers = rnd.randint(0, x.size, n_outliers)
    error[outliers] *= 10

    return y + error

# Function for computing residuals:
def resFun(c, x, y):
    return c[0] + c[1] * x + c[2] * x**2  -  y



trueCoefs = [-5, 1, 3]
sigma = 1.5
print("True coefficients: ", trueCoefs)
print("Sigma: ", sigma)

f = np.poly1d(trueCoefs)
xDat = np.linspace(0, 2, 20)
errors = sigma*np.random.normal(size=len(xDat))
yDat = f(xDat) + errors



# Initial estimate of parameters:
# x0 = np.array([1.0, 1.0, 0.0])
x0 = np.array([-4.0, 2.0, 5.0])

# Compute a standard least-squares solution:
res = least_squares(resFun, x0, args=(xDat, yDat))
#print('res: ', res)

print('Success:      ', res.success)
print('Cost:         ', res.cost)
print('Optimality:   ', res.optimality)
print('Coefficients: ', res.x)
print('Grad:         ', res.grad)
print('Residuals:    ', res.fun)

Chi2 = sum(res.fun**2)
redChi2 = Chi2/(len(xDat)-len(res.x))           # Reduced Chi^2 = Chi^2 / (n-m)
print("Chi2: ", Chi2, res.cost*2)
print("Red. Chi2: ", redChi2)




# Plot all the curves. We see that by selecting an appropriate loss we can get estimates close to
# optimal even in the presence of strong outliers. But keep in mind that generally it is recommended to try
# 'soft_l1' or 'huber' losses first (if at all necessary) as the other two options may cause difficulties in
# optimization process.

y_true    = gen_data(xDat, trueCoefs[2], trueCoefs[1], trueCoefs[0])
y_lsq     = gen_data(xDat, *res.x)

print()
#exit()


import matplotlib.pyplot as plt
#plt.style.use('dark_background')        # Invert colours
#plt.plot(xDat, yDat, 'o')
plt.errorbar(xDat, yDat, yerr=errors, fmt='ro')  # Plot red circles with actual error bars
plt.plot(xDat, y_true, 'k', linewidth=2, label='true')
plt.plot(xDat, y_lsq, label='linear loss')

plt.xlabel("t")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('scipy.optimize.least_squares.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later



