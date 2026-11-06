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

        # Now we can proceed with the actual elimination elimination
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