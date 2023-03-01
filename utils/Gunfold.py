from utils.util import np_reshape

def Gunfold(Gt, n):
    if n==1:
        return np_reshape(Gt, [Gt.shape[0], Gt.shape[1]*Gt.shape[2]])
    elif n==2:
        return np_reshape(Gt.transpose((1, 0, 2)), [Gt.shape[1], Gt.shape[0]*Gt.shape[2]])
    elif n==3:
        return np_reshape(Gt.transpose((2, 0, 1)), [Gt.shape[2], Gt.shape[0] * Gt.shape[1]])
    else:
        print("input of Gunfold is wrong")