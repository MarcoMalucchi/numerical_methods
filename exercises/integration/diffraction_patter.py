'''
The goal here is to make a plot of the diffraction pattern, which shows the modulation of the intensity of a light beam which pass trough
an aperture.

To do so we need it mathematical expression:    (J_1(kr)/kr)**2

Where J_1 is the first order Bessel function.

Then to compute the pattern, is necessary to firstly compute the Bessel function:

J_m(x) = (1/np.pi) int_0^np.pi cons(m*\theta + x*sin(\theta)) d\theta   with m = 1

So:

1. Compute the right Bessel function

2. Use it to compute the diffraction patter

3. Plot it

'''


import numpy  as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from integration.integrators import iterate_trap, iterate_simp

#############################
# GLOBAL VARIABLES.
#############################



#############################
# BESSEL FUNC. COMPUTATION.
#############################
'''
To compute it is necessary to perform an integration.

1. Define the integrand

2. Integrate it, with the correct method

'''

def integrand(theta, x, m): return np.cos(m*theta + x*np.sin(theta))

# Definition of the domain of the Bessel function (which has to already be thought as the 2D domain of the diffraction pattern).
# This is luckly 1D due to the polar rotational invariance of the Bessel function.
x  = np.arange(0, 20, 0.1)  #The domain is a square 20 x 20
y = np.arange(0, 20, 0.1)
r = np.sqrt(x**2 + y**2)
m = 1   # order of the Bessel function

# Using quad_vec function of scipy to integrate a vectorize function without using the for loop
J_1, err = integrate.quad_vec(integrand, 0, np.pi, args=(r, m))




