from utils.util import np_reshape

def Gfold(Gm, SGt, n):
    if n==1:
        return np_reshape(Gm, [SGt[0], SGt[1], SGt[2]])
    elif n==2:
        return np_reshape(Gm, [SGt[1], SGt[0], SGt[2]]).transpose((1, 0, 2))
    elif n==3:
        return np_reshape(Gm, [SGt[2], SGt[0], SGt[1]]).transpose((1, 2, 0))
    else:
        print("input of Gfold is wrong")