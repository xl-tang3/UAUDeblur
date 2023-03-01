import numpy as np

def prox_l1(b, lamda):
    return (abs(b-lamda)+b-lamda)/2+(abs(b+lamda)+b+lamda)/2