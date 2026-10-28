import numpy  as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from integrators import iterate_trap, iterate_simp

'''
FIRST EXERCISSE SSESSION IN CLASS.
Cosmological  growth factor computation.

Check the function in the file integration_tutorial.ipynb. Is pretty combersome and it is and integral function: the right extreme of the
integral is its independent variable.
Each integration is perform, then, on a interval that goes from 0 to x (a in the present case).
Best solution: for loop, the easiest one
'''

# GLOBAL VARIABLE DECLARATION.
Omega_m = [0.1, 0.3, 0.6, 1]
#Omega_m = 1

# That's just the intergrand, the factor out of the integral isnot presente here. It'll be added at the end of the computation.
def integrand(x, Omega_m):
    return 1/(x * np.sqrt(1 + Omega_m*(1/x**3 - 1)))**3

def points_1_2():

    # Definition of the domain of the function
    a = np.arange(0.1, 2, 1e-3)

    # Figure definition
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), dpi=100, sharex=True)
    ax2.set_xlabel('a')
    ax2.set_xscale('log')
    ax1.set_ylabel('D(a)')
    ax2.set_ylabel('D(a)')
    ax1.set_title('Romberg integration')
    ax2.set_title('Scipy-method quad integration')

    for Omega in Omega_m:
        # Preliminar computation of the prefactor
        Prefactor = (5/2) * Omega * np.sqrt(1 + ((1/a)**3 - 1) * Omega)

        integral_quad = []
        errors_quad = []

        integral_romb = []

        for x in a:
            value_quad, error_quad = integrate.quad(integrand, 1e-8, x, args=(Omega,))

            integral_quad.append(value_quad)
            errors_quad.append(error_quad)

            grid = np.linspace(1e-8, x, 2**10 + 1)
            values = integrand(grid, Omega)
            spacing = grid[1] - grid[0]

            value_romb = integrate.romb(values, dx=spacing)

            integral_romb.append(value_romb)

        integral_quad = np.array(integral_quad)
        integral_romb = np.array(integral_romb)

        D_quad = Prefactor * integral_quad

        D_romb = Prefactor * integral_romb

        ax2.plot(a, D_quad, label=rf"$D(a)$ for $\Omega_m = {Omega}$")
        ax1.plot(a, D_romb, label=rf"$D(a)$ for $\Omega_m = {Omega}$")

    simp_try = point_3(flag = False)
    x = 0.5*np.ones(len(simp_try))

    ax1.plot(x, simp_try,
            marker = 'o',
            markersize = 6,
            markerfacecolor = 'white',
            markeredgecolor = 'black',
            linestyle='',
            label=f'Try with simpson rule with a = 0.5')
    
    ax2.plot(x, simp_try,
            marker = 'o',
            markersize = 6,
            markerfacecolor = 'white',
            markeredgecolor = 'black',
            linestyle='',
            label=f'Try with simpson rule with a = 0.5')
    
    ax1.legend()
    ax2.legend()
    #plt.show()

def point_3(flag: bool | None = True):

    if flag:
        fig, ax = plt.subplots(1,1, figsize=(10,6), dpi=100)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.set_yscale("symlog", linthresh=1e-14)
        ax.grid()

    integ = []

    for Omega in Omega_m:

        try:
            I, errors, counter = iterate_simp(integrand, 1e-8, 0.5, 1e-12, Omega, max_iterations=25)
        except ValueError as exc:
            print(f"Invalid input or nonfinite result: {exc}")

        except TypeError as exc:
            print(f"Incorrect argument type: {exc}")

        except RuntimeError as exc:
            print(f"Integration did not converge: {exc}")

        else:

            iterations = np.arange(1, counter + 1)

            a_eval = 0.5

            prefactor = (5 / 2) * Omega * np.sqrt(
                Omega / a_eval**3 + 1 - Omega
            )

            D = prefactor * I
            integ.append(D)

            if flag:
                print(rf"Computed Grow Factor: {D}, Iterations: {counter}, Omega_m = {Omega}")

                ax.plot(iterations, errors,
                        marker='o',
                        label=rf"Error trend with iterations for $\Omega_m = {Omega}$")


    if flag:
        ax.legend()
    integ = np.array(integ)

    return integ


if __name__ == "__main__":
    points_1_2()
    point_3()
    plt.show()





