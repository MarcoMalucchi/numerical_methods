import timeit
import numpy as np

# -----------------------------------------------------------------------------
# 1. DEFINE TARGET FUNCTIONS
# -----------------------------------------------------------------------------
def line(x, m, q): 
    return x * m + q

def exp(x): 
    return np.exp(x)

# Corrected For Loop Approach (using np.arange for exact integer steps)
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

# Vectorized np.sum Approach
def trap_rule(function, n_bin, starting_point, end_point, *args):
    h = (end_point - starting_point) / n_bin
    
    # Create the coordinate grid and evaluate everything cleanly once
    x = np.linspace(starting_point, end_point, n_bin + 1)
    y = function(x, *args)
    
    I = h * ((y[0] + y[-1]) / 2 + np.sum(y[1:-1]))
    return I


# -----------------------------------------------------------------------------
# 2. BENCHMARKING ENGINE USING TIMEIT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters for testing
    bins = 1000     # Higher bin counts emphasize the speed differences dramatically
    repeats = 10000  # Number of times to run each rule to get a stable benchmark
    
    print(f"⏱️ Running benchmark: 1 function call with {bins} bins, repeated {repeats} times...\n")
    
    # We use lambda expressions to pass the target functions cleanly into timeit
    simp_time = timeit.timeit(
        stmt=lambda: simpson_rule(exp, bins, 0, 1), 
        number=repeats
    )
    
    trap_time = timeit.timeit(
        stmt=lambda: trap_rule(exp, bins, 0, 1), 
        number=repeats
    )
    
    # Calculate performance difference
    speedup = simp_time / trap_time
    
    # Print results
    print(f"Simpson Rule Function Total Time:     {simp_time:.5f} seconds")
    print(f"Trapezoidal Rule Function Total Time:  {trap_time:.5f} seconds")
    print("-" * 50)
    print(f"Here the relative speed of Simpson with respect Trapezoidal: {speedup:.1f}x TRAPEZOIDAL!")
