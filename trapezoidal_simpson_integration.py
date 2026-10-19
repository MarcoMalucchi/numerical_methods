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

    # interir_sum = 0

    # for i in indices:
    #     sum += function(starting_point + i*h, *args)

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
        n_bin *= 2  #considering the preliminar integration made above out the loop, it is important to immediatly double the number of bins
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


# Integrand functions
def line(x, m, q):
    return m*x + q


def parabola(x, r, s, t):
    return r*x**2 + s*x + t


def exponential(x):
    return np.exp(x)


def nonfinite_function(x):
    return np.full_like(x, np.nan)


def run_checks():
    accuracy = 1e-4
    passed = 0
    total = 0
    histories = []

    # Each case contains:
    # name, function, bounds, parameters, exact integral
    numerical_cases = [
        (
            "Line",
            line,
            -10, 5,
            (0.3, 5),
            0.3 * (5**2 - (-10)**2)/2 + 5 * (5 - (-10)),
        ),
        (
            "Parabola",
            parabola,
            -10, 5,
            (3.5, 0.5, 5),
            (
                3.5 * (5**3 - (-10)**3)/3
                + 0.5 * (5**2 - (-10)**2)/2
                + 5 * (5 - (-10))
            ),
        ),
        (
            "Exponential",
            exponential,
            0, 1,
            (),
            np.exp(1) - np.exp(0),
        ),
    ]

    print("NUMERICAL CHECKS")

    for name, function, a, b, parameters, exact in numerical_cases:
        for mode in ("trap", "simp"):
            total += 1
            label = f"{name} / {mode}"

            try:
                result, errors, counter = integration(
                    function, a, b, accuracy, mode, *parameters
                )

                actual_error = abs(result - exact)

                # Check both numerical accuracy and returned history.
                success = (
                    np.isfinite(result)
                    and actual_error <= accuracy
                    and len(errors) == counter
                    and counter >= 1
                    and np.all(np.isfinite(errors))
                    and np.all(errors >= 0)
                    and errors[-1] <= accuracy
                )

                passed += int(success)
                status = "PASS" if success else "FAIL"

                print(
                    f"{status}: {label:<20} "
                    f"I = {result:.10f}, "
                    f"actual error = {actual_error:.3e}, "
                    f"estimated error = {errors[-1]:.3e}, "
                    f"refinements = {counter}"
                )

                histories.append((label, errors))

            except Exception as exc:
                print(f"FAIL: {label}: {type(exc).__name__}: {exc}")

    # Each case contains:
    # description, expected exception, callable that triggers the check
    safety_cases = [
        (
            "Zero accuracy",
            ValueError,
            lambda: integration(exponential, 0, 1, 0, "trap"),
        ),
        (
            "Negative accuracy",
            ValueError,
            lambda: integration(exponential, 0, 1, -1e-4, "trap"),
        ),
        (
            "Infinite accuracy",
            ValueError,
            lambda: integration(exponential, 0, 1, np.inf, "trap"),
        ),
        (
            "NaN accuracy",
            ValueError,
            lambda: integration(exponential, 0, 1, np.nan, "trap"),
        ),
        (
            "Invalid integration mode",
            ValueError,
            lambda: integration(exponential, 0, 1, accuracy, "invalid"),
        ),
        (
            "Zero refinement limit",
            ValueError,
            lambda: integration(
                exponential, 0, 1, accuracy, "trap",
                max_iterations=0,
            ),
        ),
        (
            "Noninteger refinement limit",
            TypeError,
            lambda: integration(
                exponential, 0, 1, accuracy, "trap",
                max_iterations=2.5,
            ),
        ),
        (
            "Refinement limit reached",
            RuntimeError,
            lambda: integration(
                exponential, 0, 1, 1e-12, "trap",
                max_iterations=1,
            ),
        ),
        (
            "Zero trapezoidal bins",
            ValueError,
            lambda: trap_rule(exponential, 0, 0, 1),
        ),
        (
            "Negative Simpson bins",
            ValueError,
            lambda: simpson_rule(exponential, -2, 0, 1),
        ),
        (
            "Odd Simpson bins",
            ValueError,
            lambda: simpson_rule(exponential, 3, 0, 1),
        ),
        (
            "Noninteger trapezoidal bins",
            TypeError,
            lambda: trap_rule(exponential, 10.5, 0, 1),
        ),
        (
            "Noninteger Simpson bins",
            TypeError,
            lambda: simpson_rule(exponential, 10.0, 0, 1),
        ),
        (
            "Nonfinite integrand",
            ValueError,
            lambda: integration(
                nonfinite_function, 0, 1, accuracy, "trap"
            ),
        ),
    ]

    print("\nINPUT AND TERMINATION CHECKS")

    for description, expected_exception, operation in safety_cases:
        total += 1

        try:
            operation()
        except expected_exception:
            passed += 1
            print(f"PASS: {description}")
        except Exception as exc:
            print(
                f"FAIL: {description}: expected "
                f"{expected_exception.__name__}, got "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            print(
                f"FAIL: {description}: expected "
                f"{expected_exception.__name__}, but no exception was raised"
            )

    print(f"\nChecks passed: {passed}/{total}")

    if histories:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        for label, errors in histories:
            refinements = np.arange(1, len(errors) + 1)

            ax.plot(
                refinements,
                errors,
                marker="o",
                markersize=4,
                label=label,
            )

        # Linear near zero, logarithmic for larger errors.
        # This handles exact-zero errors without warning handling.
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.axhline(
            accuracy,
            color="black",
            linestyle="--",
            label="Requested accuracy",
        )

        ax.set_xlabel("Refinement")
        ax.set_ylabel("Richardson estimated absolute error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    run_checks()