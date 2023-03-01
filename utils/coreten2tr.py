import numpy as np
from utils.util import np_reshape

def coreten2tr(Z):
    N = len(Z)
    S = []
    for i in range(N):
        S.append(Z[i].shape[1])
    P = Z[0]
    for i in range(1,N):
        L = np_reshape(P, [np.size(P)/Z[i-1].shape[2], Z[i-1].shape[2]])
        R = np_reshape(Z[i], [Z[i].shape[0], S[i]*Z[i].shape[2]])
        P = np.matmul(L, R)

    P = np_reshape(P, [Z[0].shape[0], np.prod(S), Z[-1].shape[2]])
    P = P.transpose((1, 2, 0))
    P = np_reshape(P, [np.prod(S), Z[0].shape[0]*Z[0].shape[0]])
    temp = np.eye(Z[0].shape[0])
    temp = np_reshape(temp, [Z[0].shape[0]*Z[0].shape[0], 1])
    P = np.matmul(P, temp)
    return np_reshape(P, S)