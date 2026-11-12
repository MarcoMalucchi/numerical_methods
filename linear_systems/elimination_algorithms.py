import numpy as np
from numpy.typing import NDArray

def Gauss_elimination(A: NDArray[np.float64]) -> NDArray[np.float64]:

    B = A.copy()

    pivot_row  = 0  # To keep track of the row in which I am

    pivot_cols = [] # To keep track of the columns in which there are pivots (at the end its lenght will return the rank of the matrix)

    for col in range(B.shape[1]):

        if pivot_row == B.shape[0]: # Obviously we must not continue iterating once the last has been reached, so we exist from the loop.
            break

        where_max = pivot_row + np.argmax(abs(B[pivot_row:, col]))

        # What happens if my column has only zeros entries (in the rows I am considering)?

        if B[where_max, col] == 0:
            continue    # Stay on the same row (I don't change pivot_row) and skip to the next column
        else:
            B[[pivot_row, where_max], :] = B[[where_max, pivot_row], :] # Else proceed with the swapping

        # Now we can proceed with the actual elimination
        B[pivot_row,:] /= B[pivot_row,col]  # Note also that now this operation never returns an error since with the control above I have already
                                            # checked that B[pivot_row,col] != 0

        for j in range(pivot_row + 1, B.shape[0]):
            if B[j,col] == 0:
                continue
            else:
                B[j,:] -= B[j,col]*B[pivot_row,:]
        
        pivot_row += 1  # Finally I increment the pivot_row by one to move to the next row

        pivot_cols.append(col)

    return B, pivot_cols

def back_substitution(U, pivot_cols):
    """Return a particular solution x and a null-space basis N.

    U is the augmented echelon matrix with unit pivots.
    Every solution has the form x + N @ t.
    """
    
    n = U.shape[1] - 1

    if n in pivot_cols:
        raise ValueError("The system has no solution: a pivot is in the right-hand-side column.")

    pivot_set = set(pivot_cols)
    free_cols = [col for col in range(n) if col not in pivot_set]

    x = np.zeros(n, dtype=float)
    N = np.zeros((n, len(free_cols)), dtype=float)

    for basis_col, free_col in enumerate(free_cols):
        N[free_col, basis_col] = 1.0

    for row in range(len(pivot_cols) - 1, -1, -1):
        col = pivot_cols[row]
        coefficients = U[row, col + 1:-1]

        x[col] = U[row, -1] - coefficients @ x[col + 1:]
        N[col, :] = -(coefficients @ N[col + 1:, :])

    return x, N

def Solve(A, b) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a particular solution x and a null-space basis N for A @ x = b."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional coefficient matrix.")
    if b.shape != (A.shape[0],):
        raise ValueError("b must be a one-dimensional array with one entry per row of A.")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("A and b must contain only finite entries.")

    augmented = np.column_stack((A, b))
    U, pivot_cols = Gauss_elimination(augmented)
    
    return back_substitution(U, pivot_cols)


def LU_decomposition(A: NDArray[np.float64]) -> NDArray[np.float64]:

    U = A.copy()

    m = A.shape[0]

    L = np.eye(m , dtype = float)   # This command actually creates an mxm identity matrix
    P = np.eye(m, dtype = float)    # P is the square matrix used to keep track of the rows swappings

    pivot_row  = 0  # To keep track of the row in which I am

    pivot_cols = []

    for col in range(U.shape[1]):

        # To save the single operations of the elimination algorithm.
        L_i = np.eye(m, dtype = float)

        if pivot_row == U.shape[0]: # Obviously we must not continue iterating once the last has been reached, so we exist from the loop.
            break

        where_max = pivot_row + np.argmax(abs(U[pivot_row:, col]))

        # What happens if my column has only zeros entries (in the rows I am considering)?

        if U[where_max, col] == 0:
            continue    # Stay on the same row (I don't change pivot_row) and skip to the next column
        else:
            U[[pivot_row, where_max], :] = U[[where_max, pivot_row], :] # Else proceed with the swapping
            # Saving the swapping of the rows
            P[[pivot_row, where_max], :] = P[[where_max, pivot_row], :]
            L[[pivot_row, where_max], :pivot_row] = L[[where_max, pivot_row], :pivot_row]

        # Now we can proceed with the actual elimination
        L_i[pivot_row,pivot_row] = U[pivot_row,col]    # Saving the reascaling of the row
        U[pivot_row,:] /= U[pivot_row,col]  # Note also that now this operation never returns an error since with the control above I have already
                                            # checked that B[pivot_row,col] != 0

        for j in range(pivot_row + 1, U.shape[0]):
            if U[j,col] == 0:
                continue
            else:
                L_i[j,pivot_row] = U[j,col]   # saving the row operation
                U[j,:] -= U[j,col]*U[pivot_row,:]

        pivot_row += 1  # Finally I increment the pivot_row by one to move to the next row

        pivot_cols.append(col)

        L = L @ L_i # Updating of the general L matrix

    # print(pivot_cols, len(pivot_cols))
    # print(np.allclose(P @ A, L @ U))  # to double check

    return L, U, P

def gauss_jordan_inverse(A):
    """Return the inverse of a real square matrix using partial pivoting.

    Raise ValueError for invalid input and LinAlgError if elimination
    encounters an exactly zero pivot.
    """
    U = np.array(A, dtype=float, copy=True)

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("A must be a square two-dimensional matrix.")
    if U.shape[0] == 0:
        raise ValueError("A must have at least one row.")
    if not np.isfinite(U).all():
        raise ValueError("A must contain only finite entries.")

    n = U.shape[0]
    E = np.eye(n)

    # Forward pass: normalise each pivot and eliminate below it.
    # E records the row operations so that E @ A = U.
    for i in range(n):
        pivot_row = i + np.argmax(np.abs(U[i:, i]))
        pivot = U[pivot_row, i]

        if pivot == 0.0:
            raise np.linalg.LinAlgError(
                f"Matrix is singular: no nonzero pivot in column {i}."
            )

        U[[i, pivot_row], :] = U[[pivot_row, i], :]
        E[[i, pivot_row], :] = E[[pivot_row, i], :]

        E_i = np.eye(n)
        E_i[i, i] = 1.0 / pivot
        U[i, :] /= pivot

        for j in range(i + 1, n):
            entry = U[j, i]
            # E_i acts on the pivot row before normalisation.
            E_i[j, i] = -entry / pivot
            U[j, :] -= entry * U[i, :]

        E = E_i @ E

    # Backward pass: clear the entries above each unit pivot.
    for i in range(n - 1, 0, -1):
        E_i = np.eye(n)

        for j in range(i):
            entry = U[j, i]
            E_i[j, i] = -entry
            U[j, :] -= entry * U[i, :]

        E = E_i @ E

    return E


