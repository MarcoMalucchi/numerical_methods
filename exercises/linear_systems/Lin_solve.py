import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from scipy import linalg
from linear_systems.elimination_algorithms import Gauss_elimination, Solve, gauss_jordan_inverse

'''
PCA Principal Component Analysis.

Vector: v = [v_1, ..., v_n] of measurements

Measurements = Random variables => statistic

What can I compute?

- Variance
- Average
- Covariance or Correlation ( Pearson coefficient), relative matricies

Strongly correlated data => Information is inside a smaller subset of the data

PCA --> compression of the data to find a transformation that brings us to a smaller dimension space

Practically --> Diagonalize the covaricance matrix. Eigenvactors = new generators of the smaller space

Sort eiegenvalues in order to put the bigger one on top and smaller one lower and lower in the rows

The smaller eigenvalues bring fewer information than bigger onw, then I trow them away. Idea: you want to keep a certain fraction of the
initial covariance
'''

    # Covariance matrix from sensor measurements
C = np.array([
    [4.2, 2.1, 1.8, 0.9, 0.3, 0.2, 0.1, 0.0, -0.1, 0.0],
    [2.1, 3.8, 1.6, 0.8, 0.4, 0.1, 0.2, 0.1, 0.0, -0.1],
    [1.8, 1.6, 3.5, 1.2, 0.6, 0.3, 0.1, 0.0, 0.1, 0.0],
    [0.9, 0.8, 1.2, 2.8, 0.9, 0.4, 0.2, 0.1, 0.0, 0.1],
    [0.3, 0.4, 0.6, 0.9, 2.2, 0.7, 0.3, 0.2, 0.1, 0.0],
    [0.2, 0.1, 0.3, 0.4, 0.7, 1.8, 0.5, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.1, 0.2, 0.3, 0.5, 1.5, 0.4, 0.3, 0.2],
    [0.0, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 1.2, 0.4, 0.3],
    [-0.1, 0.0, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 0.5],
    [0.0, -0.1, 0.0, 0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.9]
])

def point_1():

    # Sample measurement in the original 10D space
    x = np.array([2.5, -1.3, 0.8, 1.2, -0.5, 0.3, -0.2, 0.4, 0.1, -0.3])

    ################################################
    # POINT 1.1 - Diagonalize the covariance matrix.
    ################################################

    w, vr = linalg.eig(C)

    #print(f'Eigenvalues: {w}\n', f'Right Eigenvectors: {vr}\n')

    ind = np.argsort(w, descending=True)

    w = w[ind]
    vr = vr[:,ind]

    print(f'Ordered indicies: {ind}\n',f'Ordered eigenvalues: {w}\n', f'Ordered eigenvectors: {vr}\n')

    ################################################
    # POINT 1.2 - Scree plot of eigenvalues.
    ################################################

    tot_var = np.sum(w)

    cum_var = np.array([])

    for k in range(w.shape[0]):
        cum_var = np.append(cum_var, (np.sum(w[:k])/tot_var)*100)

    #cum_var = np.cumsum(w)

    fig, ax = plt.subplots(1,1, figsize=(10,6), dpi=100)

    ax.axhline(90, linestyle='--')
    ax.axhline(95, linestyle='--')
    u = np.arange(w.shape[0])
    ax.plot(u[1:], cum_var[1:], marker='o')

    ################################################
    # POINT 1.3 - Projection onto the PC basis.
    ################################################

    z = np.transpose(vr) @ x

    print(f'Projection onto PC basis: {z}\n')

    ##########################################################
    # POINT 1.4 - Dimensionality Reduction and Reconstruction.
    ##########################################################

    z_3 = z[:3]
    x_3 = vr[:, :3] @ z_3

    print(f'Approximation of original measurement: {x_3}\n', f'Orginal measurement: {x}\n')

    recon_error = np.linalg.norm(x - x_3)

    print(f'Reconstruction error: {recon_error}\n')

    return

def point_2():
    b = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2])
    a = linalg.solve(C, b)
    print(f'Solution vector: {a}\n')
    print(f'Is it right?: {np.allclose(C @ a, b)}\n')

    # Now with my functions

    a, N = Solve(C, b)
    print(f'Solution vector: {a}\n',f'And the null-space basis: {N}\n')
    print(f'Is it right?: {np.allclose(C @ a, b)}\n')

    inverse = gauss_jordan_inverse(C)
    print(f'Inverse of C: {inverse}\n')
    print(f'Is it right?: {np.allclose(inverse @ C, np.eye(10))}\n')

    return

def point_3():
    # Solving an electric circuit with linear algerba, we will use complex notation to taking into account the alternate current
    R = np.ones(6)
    C = np.ones(2)
    omega = 1
    t = np.linspace(0, 1, 100)
    Vcc = 5*np.exp(1j*omega*t)
    Vgnd = 0

    A = np.array([
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

    return

if __name__ == "__main__":
    point_1()
    point_2()
    plt.show()


