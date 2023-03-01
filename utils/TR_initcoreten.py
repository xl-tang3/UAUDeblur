import numpy as np

def TR_initcoreten(S, r):
    N = len(S)
    Z = []
    for i in range(N-1):
        Z.append(np.random.randn(r[i], S[i], r[i+1]))
    Z.append(np.random.randn(r[-1], S[-1], r[0]))

    return Z