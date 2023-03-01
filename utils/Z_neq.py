from utils.util import np_reshape
import numpy as np

def Z_neq(Z, n):
    N = len(Z)
    P = Z[(n)%N]
    for i in range(n, n+N-2):
        print(type(P), i)
        zl = np_reshape(P, (P.shape[0]*P.shape[1], P.shape[2]))
        zr = np_reshape(Z[(i+1)%N], (Z[(i+1)%N].shape[0], Z[(i+1)%N].shape[1]*Z[(i+1)%N].shape[2]))
        P = np.matmul(zl, zr)
    return np_reshape(P, (Z[(n)%N].shape[0], P.size/(Z[(n)%N].shape[0]*Z[(n+N-2)%N].shape[2]), Z[(n+N-2)%N].shape[2]))
