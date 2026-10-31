import numpy as np
from numpy.typing import NDArray

def Gauss_elimination(A: NDArray[np.float64]) -> NDArray[np.float64]:
    C = np.array(A, dtype=float, copy=True) # Convert lists and int type arrays in float arrays

    if C.ndim != 2:
        raise ValueError("A must be two-dimensional.")

    if C.shape[0] != C.shape[1]:
        raise ValueError("A must be square.")

    for i in range(C.shape[0]):
        # Let's start ordering the columns
        where_max = i + np.argmax(abs(C[i:, i]))

        if C[where_max, i] == 0:
            raise ValueError("Singular matrix: no nonzero pivot available.")

        C[[i, where_max], :] = C[[where_max, i], :]

        # Now we can proceed with the elimination
        C[i,:] /= C[i,i]

        for j in range(i + 1, C.shape[0]):
            if C[j,i] == 0:
                continue
            else:
                C[j,:] -= C[j,i]*C[i,:]
        
        return C