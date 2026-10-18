'''
TRAPEZOIDAL and SIMPSON INTEGRATION SCRIPT.
 
- Trapezoidal rule:
    We try to approximate the area under a function with a set of trapezoids. The domain is divided in subintervals and the function
    in each interval is approximated with a linear function, i.e. linearized.
    Then, summing the area of each trapezoid build as above, we find the formula:

        I = h * [f(a)/2 + f(b)/2 + sum_k=1^N-1 f(a + k*h)]

    where N is the number of subintervals and h = (b-a)/N.
    With trapezoidal rule the error commited is O(h^2) at the first order.

- Simpson rule:
    Instead of linearizing the function in the subintervals, we approximate it with a quadratic function, convering two subintervals,
    instead of just one. Then the furmula is:

        I = h/3 * [f(a) + f(b) + 4*sum_k=1^N-1 (ODD) f(a + k*h) + 2*sum_k=1^N-1 (EVEN) f(a + k*h)]

    note the sums diveded between even and odd values of the index.
    With Simpson rule the error commited is O(h^4) at the first order.

'''

import numpy as np

# This function has the only task to compute the trapezoidal rule for a given function, a given number of bins, onto the function domain
def trap_rule(function, n_bin, starting_poin, end_point, *args):
#NOTE: the "*args" parameter is needed to handle the parameters of the integrand funtion: in principle I don't know how many paramenters
#the function will have, so I use *args to prepear my trap_rule function to accept any paramenters I'll pass to it (take a look to  **kwargs)
#for something even more intresting, so to have a complitely custumizable function.

    h = (end_point - starting_poin)/n_bin

    f_a = function(starting_poin, *args)
    f_b = function(end_point, *args)

    # indices = np.arange(1, n_bin)

    # sum = 0

    # for i in indices:
    #     sum += function(starting_poin + i*h, *args)

    x = np.linspace(starting_poin, end_point, n_bin + 1)
    sum = np.sum(function(x, *args)[1:-1])

    I = h*(f_a/2 + f_b/2 + sum)

    return I

# Integrand functions
def line(x, m, q): return x*m + q

# Parabola
def parabola(x, r, s, t): return r*x**2 + s*x + t

## EXECUTION OF THE TRAP_RULE FUNCTION

# Definition of relevant paramenters
a = -10
b = 5
n_bins = 800

# Integrand function paramenters
m_param = 1
q_param = 0

r_param = 3.5
s_param = .5
t_param = 5

# Exact integral
I = r_param*(b**3/3 - a**3/3) + s_param*(b**2/2 - + a**2/2) + t_param*(b - a)

# Numerical integral
I_num = trap_rule(parabola, n_bins, a, b, r_param, s_param, t_param)

print(f'Numerical Intregral: {I_num}\nExact integral: {I}\nError: {((I_num - I)/I)*100:.4f}%')