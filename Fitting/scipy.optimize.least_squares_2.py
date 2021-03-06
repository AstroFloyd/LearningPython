#!/bin/env python3

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

"""Solve a curve fitting problem using robust loss function to take care of outliers in the data. Define the
model function as y = a + b * exp(c * t), where t is a predictor variable, y is an observation and a, b, c are
parameters to estimate.
"""

import numpy as np
from scipy.optimize import least_squares


# Function which generates the data with noise and outliers:
def gen_data(t, a, b, c, noise=0, n_outliers=0, random_state=0):
    y = a + b * np.exp(t * c)
    
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 10

    return y + error

# Function for computing residuals:
def fun(x, t, y):
    return x[0] + x[1] * np.exp(x[2] * t) - y


# Define the model parameters:
a = 0.5
b = 2.0
c = -1
t_min = 0
t_max = 10
n_points = 15

# Generate the data:
t_train = np.linspace(t_min, t_max, n_points)
y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)


# Initial estimate of parameters:
x0 = np.array([1.0, 1.0, 0.0])

# Compute a standard least-squares solution:
res = least_squares(fun, x0, args=(t_train, y_train))
#print('res: ', res)

print('Success:      ', res.success)
print('Cost:         ', res.cost)
print('Optimality:   ', res.optimality)
print('Coefficients: ', res.x)
print('Grad:         ', res.grad)
print('Residuals:    ', res.fun)


# Plot all the curves. We see that by selecting an appropriate loss we can get estimates close to
# optimal even in the presence of strong outliers. But keep in mind that generally it is recommended to try
# 'soft_l1' or 'huber' losses first (if at all necessary) as the other two options may cause difficulties in
# optimization process.

t_test    = np.linspace(t_min, t_max, n_points * 10)
y_true    = gen_data(t_test, a, b, c)
y_lsq     = gen_data(t_test, *res.x)


import matplotlib.pyplot as plt
#plt.style.use('dark_background')        # Invert colours
plt.plot(t_train, y_train, 'o')
plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
plt.plot(t_test, y_lsq, label='linear loss')

plt.xlabel("t")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('scipy.optimize.least_squares_2.png')                 # Save the plot as png
plt.close()                             # Close the plot in order to start a new one later


print()

