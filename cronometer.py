import timeit
import numpy as np

# -----------------------------------------------------------------------------
# 1. DEFINE TARGET FUNCTIONS
# -----------------------------------------------------------------------------
def line(x, m, q): 
    return x * m + q

# Corrected For Loop Approach (using np.arange for exact integer steps)
def trap_rule_loop(function, n_bin, starting_point, end_point, *args):
    h = (end_point - starting_point) / n_bin
    
    f_a = function(starting_point, *args)
    f_b = function(end_point, *args)
    
    # Generates exact integer intervals from 1 to n_bin-1
    indices = np.arange(1, n_bin)
    
    interior_sum = 0
    for i in indices:
        interior_sum += function(starting_point + i * h, *args)
        
    I = h * (f_a / 2 + f_b / 2 + interior_sum)
    return I

# Vectorized np.sum Approach
def trap_rule_vec(function, n_bin, starting_point, end_point, *args):
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
    repeats = 1000  # Number of times to run each rule to get a stable benchmark
    
    print(f"⏱️ Running benchmark: 1 function call with {bins} bins, repeated {repeats} times...\n")
    
    # We use lambda expressions to pass the target functions cleanly into timeit
    loop_time = timeit.timeit(
        stmt=lambda: trap_rule_loop(line, bins, 1, 20, 1, 0), 
        number=repeats
    )
    
    vec_time = timeit.timeit(
        stmt=lambda: trap_rule_vec(line, bins, 1, 20, 1, 0), 
        number=repeats
    )
    
    # Calculate performance difference
    speedup = loop_time / vec_time
    
    # Print results
    print(f"🔄 For-Loop Total Time:     {loop_time:.5f} seconds")
    print(f"🚀 Vectorized np.sum Time:  {vec_time:.5f} seconds")
    print("-" * 50)
    print(f"🎯 The Vectorized approach is roughly {speedup:.1f}x FASTER!")
