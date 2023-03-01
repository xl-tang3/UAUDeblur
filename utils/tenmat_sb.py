import numpy as np
from utils.util import np_reshape

def tenmat_sb(X, k):
    S = X.shape
    N = np.size(S)
    if k==1:
        return np_reshape(X, [S[0], np.size(X)/S[0]])
    elif k==N:
        return np_reshape(X, [np.size(X)/S[N-1], S[N-1]]).transpose()
    else:
        X = np_reshape(X, [np.prod(S[:k-1]), np.size(X)/np.prod(S[:k-1])])
        X = X.transpose()
        return np_reshape(X, [S[k-1], np.size(X)/S[k-1]])