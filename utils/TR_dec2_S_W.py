import numpy as np
from .tenmat_sb import tenmat_sb
from .Z_neq import Z_neq
from .Gfold import Gfold
from .Gunfold import Gunfold
from .prox_l1 import prox_l1
from .coreten2tr import coreten2tr

def TR_dec2_S_W(X_TR,Lamda,Maxiter,rho,u,S_k,G1,G2,G3,Y):
    N = X_TR.ndim
    G = []
    print(type(G1))
    G.append(G1)
    G.append(G2)
    G.append(G3)
    iter = 0
    while iter < Maxiter:
        iter = iter + 1
        for n in range(1, 1+N):
            Q = tenmat_sb(Z_neq(G, n), 2)
            Q = Q.transpose()
            temp1 = np.matmul(tenmat_sb(X_TR, n), Q.transpose())
            temp2 = np.matmul(Q, Q.transpose())
            G[n-1] = Gfold(np.matmul((2*Lamda*temp1+rho*Gunfold(G[n-1], 2)),np.linalg.pinv((2*Lamda*temp2+rho*np.eye(Q.shape[0])))), G[n-1].shape, 2)
        W = 1/(np.abs((2*(Y-X_TR)+rho*S_k)/(2+rho))+np.spacing(1))
        S_k = prox_l1((2*(Y-X_TR)+rho*S_k)/(rho+2), W*u/(2+rho))
        X_TR = coreten2tr(G)
    G1 = G[0]
    G2 = [G1]
    G3 = G[2]
    return [X_TR,S_k,G1,G2,G3]
