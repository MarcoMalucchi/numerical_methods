"""Checks for integrators.py. Run: python check_integrators.py [--plot]

Place beside integrators.py. Assumes vectorised real-valued integrands,
absolute Richardson tolerance, initial 10 intervals, and keyword-only
max_iterations. No third-party test framework is required.
"""
import argparse
from functools import partial
import sys
import numpy as np
from integrators import (
    validate_controls, trap_rule, simpson_rule, iterate_trap, iterate_simp,
)


def line(x, m, q):
    return m * x + q


def parabola(x, r, s, t):
    return r * x**2 + s * x + t


def nonfinite(x):
    return np.full_like(x, np.nan, dtype=float)


def nonfinite_on_refinement(x):
    # Initial grid contains 0, .1, ..., 1; first new grid contains .05.
    return np.where(np.isclose(x, 0.05, rtol=0, atol=1e-14), np.nan, np.exp(x))


class CountedFunction:
    def __init__(self, function):
        self.function = function
        self.samples = 0

    def __call__(self, x, *args):
        self.samples += np.size(x)
        return self.function(x, *args)


def run_checks(plot=False):
    passed = 0
    total = 0
    histories = []
    tolerance = 1e-4

    def check(label, operation, expected_exception=None):
        nonlocal passed, total
        total += 1
        try:
            outcome = operation()
        except Exception as exc:
            success = (
                expected_exception is not None
                and isinstance(exc, expected_exception)
            )
            detail = f'{type(exc).__name__}: {exc}'
        else:
            success = expected_exception is None and bool(outcome)
            detail = ('condition satisfied' if success else
                      f'expected {expected_exception.__name__}'
                      if expected_exception else 'condition not satisfied')
        passed += int(success)
        print(f'{"PASS" if success else "FAIL"}: {label}: {detail}')

    print('CONTROL VALIDATION')
    check('valid controls', lambda: validate_controls(tolerance, 15) is None)
    for value in (0, -1e-4, np.inf, -np.inf, np.nan):
        check(f'accuracy={value}', partial(validate_controls, value, 15), ValueError)
    for value in (0, -1):
        check(f'limit={value}', partial(validate_controls, tolerance, value), ValueError)
    for value in (2.5, 2.0, True, '2', None):
        check(f'limit={value!r}', partial(validate_controls, tolerance, value), TypeError)

    print('\nFIXED-GRID RULES')
    for name, rule in (('trap', trap_rule), ('simp', simpson_rule)):
        for n in (0, -2):
            check(f'{name}: invalid bins {n}', partial(rule, np.exp, n, 0, 1), ValueError)
        for n in (10.5, 10.0, True, '10', None):
            check(f'{name}: invalid bin type {n!r}', partial(rule, np.exp, n, 0, 1), TypeError)
        check(f'{name}: line with parameters',
              lambda rule=rule: np.isclose(rule(line, 10, -10, 5, .3, 5), 63.75,
                                           rtol=0, atol=1e-12))
        check(f'{name}: reversed bounds',
              lambda rule=rule: np.isclose(rule(np.exp, 20, 1, 0),
                                           -rule(np.exp, 20, 0, 1), rtol=0, atol=1e-12))
        check(f'{name}: equal bounds', lambda rule=rule: rule(np.exp, 10, 1, 1) == 0)
        def convergence(rule=rule, name=name):
            exact = np.expm1(1)
            coarse = abs(rule(np.exp, 10, 0, 1) - exact)
            fine = abs(rule(np.exp, 20, 0, 1) - exact)
            return np.isclose(coarse / fine, 4 if name == 'trap' else 16, rtol=.02)
        check(f'{name}: expected convergence order on exp', convergence)
    check('simp: odd bins', partial(simpson_rule, np.exp, 3, 0, 1), ValueError)
    check('simp: cubic exactness',
          lambda: np.isclose(simpson_rule(lambda x: x**3, 10, 0, 1), .25,
                             rtol=0, atol=1e-14))

    print('\nITERATIVE NUMERICAL CHECKS')
    cases = [
        ('line / parameters', line, -10, 5, (.3, 5), 63.75),
        ('parabola / parameters', parabola, -10, 5, (3.5, .5, 5), 1368.75),
        ('exponential', np.exp, 0, 1, (), np.expm1(1)),
        ('reversed bounds', np.exp, 1, 0, (), -np.expm1(1)),
        ('equal bounds', np.exp, 1, 1, (), 0),
        ('zero integral', lambda x: x, -1, 1, (), 0),
        ('constant array', lambda x: np.full_like(x, 3.), -2, 4, (), 18),
        ('negative integral', lambda x: -np.exp(x), 0, 1, (), -np.expm1(1)),
    ]
    for mode, integrate, fixed_rule, denominator in (
        ('trap', iterate_trap, trap_rule, 3),
        ('simp', iterate_simp, simpson_rule, 15),
    ):
        for label, function, a, b, parameters, exact in cases:
            def numerical(function=function, a=a, b=b, parameters=parameters,
                          exact=exact, label=label, integrate=integrate, mode=mode):
                result, errors, counter = integrate(function, a, b, tolerance, *parameters)
                errors = np.asarray(errors)
                valid = (
                    type(counter) is int and 1 <= counter <= 15
                    and errors.ndim == 1 and errors.size == counter
                    and np.isfinite(result) and np.all(np.isfinite(errors))
                    and np.all(errors >= 0) and errors[-1] <= tolerance
                    and abs(result - exact) <= tolerance
                )
                if valid:
                    histories.append((f'{mode}: {label}', errors))
                return valid
            check(f'{mode}: {label}', numerical)

        def reuse(integrate=integrate, fixed_rule=fixed_rule, denominator=denominator):
            counted = CountedFunction(np.exp)
            result, errors, counter = integrate(counted, 0, 1, 1e-10)
            final_n = 10 * 2**counter
            expected_history = []
            previous = fixed_rule(np.exp, 10, 0, 1)
            for k in range(1, counter + 1):
                current = fixed_rule(np.exp, 10 * 2**k, 0, 1)
                expected_history.append(abs(current - previous) / denominator)
                previous = current
            return (
                counted.samples == final_n + 1
                and np.isclose(result, previous, rtol=1e-13, atol=1e-13)
                and np.allclose(errors, expected_history, rtol=1e-6, atol=1e-13)
            )
        check(f'{mode}: reuse, final grid and Richardson history', reuse)
        check(f'{mode}: convergence permitted at limit',
              lambda integrate=integrate: integrate(line, 0, 1, tolerance, 2., 3.,
                                                    max_iterations=1)[2] == 1)

        print(f'\n{mode.upper()} INPUT AND TERMINATION CHECKS')
        for value in (0, -1e-4, np.inf, -np.inf, np.nan):
            check(f'{mode}: accuracy={value}',
                  partial(integrate, np.exp, 0, 1, value), ValueError)
        for value in (0, -1):
            check(f'{mode}: limit={value}',
                  partial(integrate, np.exp, 0, 1, tolerance, max_iterations=value), ValueError)
        for value in (2.5, 2.0, True, '2', None):
            check(f'{mode}: limit={value!r}',
                  partial(integrate, np.exp, 0, 1, tolerance, max_iterations=value), TypeError)
        check(f'{mode}: refinement limit reached',
              partial(integrate, np.exp, 0, 1, 1e-12, max_iterations=1), RuntimeError)
        check(f'{mode}: initial nonfinite result',
              partial(integrate, nonfinite, 0, 1, tolerance), ValueError)
        check(f'{mode}: nonfinite result during refinement',
              partial(integrate, nonfinite_on_refinement, 0, 1, tolerance), ValueError)

    print(f'\nChecks passed: {passed}/{total}')
    print('Known-answer checks support these examples; Richardson estimates are not universal bounds.')
    if plot and histories:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, errors in histories:
            ax.plot(np.arange(1, len(errors) + 1), errors, marker='o', label=label)
        ax.set_yscale('symlog', linthresh=1e-12)
        ax.axhline(tolerance, color='black', linestyle='--', label='Tolerance')
        ax.set_xlabel('Refinement')
        ax.set_ylabel('Richardson estimated absolute error')
        ax.grid(True, alpha=.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        plt.show()
    return passed == total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true', help='Plot error histories')
    options = parser.parse_args()
    sys.exit(0 if run_checks(options.plot) else 1)
