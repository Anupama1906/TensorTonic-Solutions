import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    row,col = len(A),len(A[0])
    A_T = np.full((col,row),0)
    for i in range(row):
        for j in range(col):
            A_T[j][i] = A[i][j]
    return A_T
