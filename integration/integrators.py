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

3. Reduce the number of logical controls inside a function as much as possible. Doing your best to keep the code readable

'''

import numpy as np
import matplotlib.pyplot as plt
import warnings

import numpy as np

# To check if the inserted control paramenters are valid (this function is separated from the other since these a checkes needed for all of them)
def validate_controls(accuracy, max_iterations):
    if not np.isfinite(accuracy) or accuracy <= 0:
        raise ValueError("accuracy must be finite and strictly positive.")

    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an int.")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be strictly positive.")

# This function has the only task to compute the trapezoidal rule for a given function, a given number of bins, onto the function domain
def trap_rule(function, n_bin, starting_point, end_point, *args):
#NOTE: the "*args" parameter is needed to handle the parameters of the integrand funtion: in principle I don't know how many paramenters
#the function will have, so I use *args to prepear my trap_rule function to accept any paramenters I'll pass to it (take a look to  **kwargs)
#for something even more intresting, so to have a complitely custumizable function.

    if type(n_bin) == int:
        if n_bin <= 0:
            raise ValueError(f"Selected number of bin {n_bin}, not valid. It has to be positive.")
    else:
        raise TypeError(f"The type of n_bin: {n_bin}, is wrong. It must be an int.")

    h = (end_point - starting_point)/n_bin

    x = np.linspace(starting_point, end_point, n_bin + 1)
    y = function(x, *args)

    # indices = np.arange(1, n_bin)

    # interior_sum = 0

    # for i in indices:
    #     interior_sum += function(starting_point + i*h, *args)

    interior_sum = np.sum(y[1:-1])

    I = h*((y[0] + y[-1])/2 + interior_sum)

    return I

# This function, instead use the Simpson rule
def simpson_rule(function, n_bin, starting_point, end_point, *args):

    if type(n_bin) == int:
        if n_bin % 2 != 0 or n_bin <= 0:
            raise ValueError(f"Selected number of bins: {n_bin} not valid. It has to be both positive and even.")
    else:
        raise TypeError(f"The type of n_bin: {n_bin}, is wrong. It must be an int.")
        
    x = np.linspace(starting_point, end_point, n_bin + 1)
    y = function(x, *args)

    h = (end_point - starting_point)/n_bin

    odd_sum = np.sum(y[1:-1:2])
    even_sum = np.sum(y[2:-1:2])

    I = h/3 * (y[0] + y[-1] + 4*odd_sum + 2*even_sum)

    return I

# This one implement an iteration of both trapezoidal and Simpson rules together with the error computation, to check for convergence
def integration(function, starting_point, end_point, accuracy, INT_MODE: str, *args, max_iterations = 15):
#NOTE: parameters after *args has to pass specifing their names.

    if not np.isfinite(accuracy) or accuracy <= 0:
        raise ValueError("accuracy must be finite and strictly positive.")

    # max_iterations serves as a limit on the number o fiteration the while loop can actaully do to avoid infinite loops and saturation of the memory
    if type(max_iterations) is not int:
        raise TypeError("max_iterations must be an int.")   # Diferent errors has to be handle separately

    if max_iterations <= 0:
        raise ValueError("max_iterations must be strictly positive.")

    if INT_MODE == "trap":
        rule = trap_rule
        denominator = 3 # denominator of the error-computation formula, which depends upon the chosen rule
    elif INT_MODE == "simp":
        rule = simpson_rule
        denominator = 15
    else:
        raise ValueError(f"Invalid mode: {INT_MODE}. Choose either 'trap' or 'simp'.")

    error = float('inf')    # initialize to infinity so that the while loop is actually started (if it would zero the while would never start)
    counter = 0
    errors = [] # declaire as an empty list because np.append creates everytime a new array, then it is slower then append to a list,
                # which just change the already allocated list
    n_bin = 10   #nominal starting number of bins of the first cycle of integration
    previous_integration = rule(function, n_bin, starting_point, end_point, *args)  # computating the fisrt integration before starting the
                                                                                    # loop is better for what concern the error computation
                                                                                    # which, now, can be initialize directly in the loop
    if not np.isfinite(previous_integration):
        raise ValueError("The initial integration result is not finite.")
    
    current_integration = 0

    while error > accuracy:
        if counter >= max_iterations:
                raise RuntimeError(
                    f"Requested accuracy was not reached after {counter} refinements. "
                    f"Last estimated error: {error}"
                )
        n_bin *= 2  #considering the preliminar integration made above, out of the loop, it is important to immediatly double the number of bins
                    #as first operation in the loop, otherwise the first cycle will return a null error.
        current_integration = rule(function, n_bin, starting_point, end_point, *args)
        error = abs(current_integration - previous_integration)/denominator
        if not np.isfinite(error):
            raise ValueError(
                "The error estimate is not finite. "
                "Check the integrand and integration bounds."
            )
        counter += 1
        errors.append(error)
        previous_integration = current_integration
        
            

    I = current_integration
    errors = np.array(errors)

    return I, errors, counter


def iterate_trap(function, a, b, accuracy, *args, max_iterations = 15):

    validate_controls(accuracy, max_iterations)

    n_bin = 10
    h = (b - a)/n_bin
    
    y = function(np.linspace(a, b, n_bin + 1), *args)

    previous = h * ((y[0] + y[-1]) / 2 + np.sum(y[1:-1]))

    if not np.isfinite(previous):
        raise ValueError("The initial integration is not finite.")

    errors = []

    error = float('inf')
    counter = 0

    while error > accuracy:
        if counter >= max_iterations:
            raise RuntimeError(
                f"Requested accuracy was not reached after {counter} refinements. "
                f"Last estimated error: {error}"
            )

        # h and n still describe the coarser grid.
        midpoints = a + (np.arange(n_bin) + 0.5)*h
        midpoints_sum = np.sum(function(midpoints, *args))
        
        current = previous/2 + 0.5*h*midpoints_sum
        error = abs(current - previous)/3

        if not np.isfinite(error) or not np.isfinite(current):
            raise ValueError(f"A nonfinite integration or error occurred: {error}, {current}")
        
        counter += 1
        errors.append(error)
        previous = current
        n_bin *= 2
        h *= 0.5

    I = current
    errors = np.array(errors)

    return I, errors, counter

def iterate_simp(function, a, b, accuracy, *args, max_iterations = 15):

    validate_controls(accuracy, max_iterations)

    n_bin = 10
    h = (b-a)/n_bin

    y = function(np.linspace(a, b, n_bin + 1), *args)

    previous_odd_sum = np.sum(y[1:-1:2])
    previous_even_sum = np.sum(y[2:-1:2])

    previous = h/3 * (y[0] + y[-1] + 4*previous_odd_sum + 2*previous_even_sum)

    endpoint_sum = y[0] + y[-1]

    if not np.isfinite(previous):
        raise ValueError("The initial integration is not finite.")

    errors = []

    error = float('inf')
    counter = 0

    current = 0
    current_odd_sum = 0
    current_even_sum = 0
    
    while error > accuracy:
        if counter >= max_iterations:
                raise RuntimeError(
                    f"Requested accuracy was not reached after {counter} refinements. "
                    f"Last estimated error: {error}"
                )

        # All old interior nodes become even-indexed nodes.
        current_even_sum = previous_even_sum + previous_odd_sum

        # New midpoints become odd-indexed nodes.
        midpoints = a + (np.arange(n_bin) + 0.5) * h
        current_odd_sum = np.sum(function(midpoints, *args))

        # Simpson's formula now uses the finer spacing.
        h *= .5

        current = h / 3 * (endpoint_sum + 4 * current_odd_sum + 2 * current_even_sum)

        error = abs(current - previous)/15

        if not np.isfinite(error) or not np.isfinite(current):
            raise ValueError(f"A nonfinite integration or error occurred: {error}, {current}")
        
        counter += 1
        errors.append(error)
        previous = current
        previous_odd_sum = current_odd_sum
        previous_even_sum = current_even_sum
        n_bin *= 2

    I = current
    errors = np.array(errors)

    return I, errors, counter

