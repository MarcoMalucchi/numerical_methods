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

'''
GOOD-BEHAVE RULES FOR HIGHER PERFORMANCES

1. Make the integrating functions evaluate the function on its domain as few times as possible.
    It is a long task, even longer if n_bin is particularly high and the function is pretty complex

2. Put every computation inside a one-kilometer long line DO NOT INCREASE THE SPEED OF YOUR CODE. Instead use the variables as placeholders
when needed, so to enhance the readability and the clarity of the script!

'''

import numpy as np
import matplotlib.pyplot as plt
import warnings

# This function has the only task to compute the trapezoidal rule for a given function, a given number of bins, onto the function domain
def trap_rule(function, n_bin, starting_poin, end_point, *args):
#NOTE: the "*args" parameter is needed to handle the parameters of the integrand funtion: in principle I don't know how many paramenters
#the function will have, so I use *args to prepear my trap_rule function to accept any paramenters I'll pass to it (take a look to  **kwargs)
#for something even more intresting, so to have a complitely custumizable function.

    h = (end_point - starting_poin)/n_bin

    x = np.linspace(starting_poin, end_point, n_bin + 1)
    y = function(x, *args)

    # indices = np.arange(1, n_bin)

    # sum = 0

    # for i in indices:
    #     sum += function(starting_poin + i*h, *args)

    sum = np.sum(y[1:-1])

    I = h*((y[0] + y[-1])/2 + sum)

    return I

def simpson_rule(function, n_bin, starting_poin, end_point, *args):

    if n_bin % 2 != 0:
        raise ValueError("Simpson's Rule requires an EVEN number of bins (n_bins).")
    
    x = np.linspace(starting_poin, end_point, n_bin + 1)
    y = function(x, *args)

    h = (end_point - starting_poin)/n_bin

    odd_sum = np.sum(y[1:-1:2])
    even_sum = np.sum(y[2:-1:2])

    I = h/3 * (y[0] + y[-1] + 4*odd_sum + 2*even_sum)

    return I

def integration(function, starting_point, end_point, accuracy, INT_MODE: str, *args):

    allowed_int_mode = {"trap", "simp"}

    previous_integration = 0
    current_integration = 0
    error = float('inf')    #Start with infinite error so the while loop is forced to execute the first time, otherwise if error = 0 the
                            # while loop will never start and the function will return only 0 and empty arrays
    counter = 0
    errors = np.array([])
    n_bin = 10   #nominal starting number of bins of the first cycle of integration

    if INT_MODE not in allowed_int_mode:
        raise ValueError(f"Invalid mode: {INT_MODE}. Choose either 'trap' or 'simp'.")
    elif INT_MODE == "trap":
        while error > accuracy:
            current_integration = trap_rule(function, n_bin, starting_point, end_point, *args)
            error = abs(current_integration - previous_integration)/3
            counter += 1
            if counter > 1:
                errors = np.append(errors, error)   # To disregard the first computed error, since it is equal to first-iteration integral
            previous_integration = current_integration
            n_bin *= 2
            
            

    elif INT_MODE == "simp":
        while error > accuracy:
            current_integration = simpson_rule(function, n_bin, starting_point, end_point, *args)
            error = abs(current_integration - previous_integration)/15
            counter += 1
            if counter > 1:
                errors = np.append(errors, error)   # To disregard the first computed error, since it is equal to first-iteration integral
            previous_integration = current_integration
            n_bin *= 2

    I = current_integration

    return I, errors, counter


# Integrand functions
def line(x, m, q): return x*m + q

# Parabola
def parabola(x, r, s, t): return r*x**2 + s*x + t

## EXECUTION OF THE NUMERICAL INTEGRATION

# Definition of relevant paramenters
a = -10
b = 5
n_bins = 800    #It MUST BE even
accuracy = 1e-4

# Integrand function paramenters
m_param = .3
q_param = 5

r_param = 3.5
s_param = .5
t_param = 5

# Exact integral
I_l = m_param*(b**2/2 - a**2/2) + q_param*(b - a)

I_p = r_param*(b**3/3 - a**3/3) + s_param*(b**2/2 - + a**2/2) + t_param*(b - a)

# Numerical integral
I_num = simpson_rule(parabola, n_bins, a, b, r_param, s_param, t_param)

I, errors, counter = integration(parabola, a, b, accuracy, "trap", r_param, s_param, t_param)

print(f'Numerical Intregral with Simpson: {I_num}\nExact integral: {I_p}\nError: {(abs(I_num - I_p)/I_p)*100:.4f}%\n')

print(f'Numerical Intregral with iterations: {I}\n',
      f'Exact integral: {I_p}\n',
      f'Error: {(abs(I_num - I_p)/I_p)*100:.6f}%\n',
      f'Iterations: {counter}')

iteration = np.linspace(2, counter, counter-1)

fig, ax = plt.subplots(figsize = (10,6), dpi=100)

ax.plot(
    iteration,
    errors,
    linestyle='--',
    color='lightgrey',
    linewidth=2,
    marker='o',
    markersize=8,
    markerfacecolor='red',   # Inside of the marker is RED
    markeredgecolor='black',  # Border of the marker is BLACK
    label='Errors trend'
)

# Try-except block using the warning filter context manager
try:
    with warnings.catch_warnings():
        # Treat all UserWarnings inside this block as explicit Exceptions
        warnings.simplefilter("error", UserWarning)
        
        plt.yscale('log')  # This line will trigger the exception if a 0 is present

except UserWarning:
    print("⚠️ Caught a log-scale zero warning! Switching to 'symlog' scale automatically.")
    # Fallback plan: symlog handles exact zeros perfectly without warning
    plt.yscale('symlog')

ax.set_xlabel("Iteration")
ax.set_ylabel("Richardson Estimated Error")
ax.legend()
plt.show()
