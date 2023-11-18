import torch
import torch.nn as nn
from torch.nn import init

def my_reshape(input, L1, L2):
    '''
    input: a tensor of n_1xn_2xn3...xn_m
    shape: a list of int
    return: a tensor of shape
    '''
    # X = Z1.permute(2, 1, 0).reshape(25 * 3, 4).transpose(1, 0) 三维度

    ans = input.permute(2, 1, 0).reshape(L2, L1).transpose(1, 0)
    return ans

def tr_product(Z1, Z2, Z3):
    '''
    Z: list of tr factor 3-way
    return: a tensor
    '''
    N = 3
    p = Z1
    S_mm = Z1.shape[1] * Z2.shape[1] * Z3.shape[1]

    L = my_reshape(p, int(torch.numel(p) / (Z1.shape[2])), Z1.shape[2])
    R = my_reshape(Z2, Z2.shape[0], Z2.shape[1]*Z2.shape[2])
    p = torch.mm(L, R)

    L = p.permute(1, 0).reshape(Z2.shape[2], int(torch.numel(p) / (Z2.shape[2]))).transpose(1, 0)
    R = my_reshape(Z3, Z3.shape[0], Z3.shape[1] * Z3.shape[2])
    p = torch.mm(L, R)

    p = p.transpose(1, 0).reshape([Z3.shape[2], S_mm, Z1.shape[0]]).permute(1, 0, 2)
    p = my_reshape(p, S_mm, Z1.shape[0] * Z1.shape[0])
    temp = torch.eye(Z1.shape[0], Z1.shape[0]).type(dtype).cuda()
    temp = temp.reshape(Z1.shape[0] * Z1.shape[0], 1)
    p = torch.mm(p, temp)
    X = p.transpose(1, 0).reshape(Z3.shape[1], Z2.shape[1], Z1.shape[1]).permute(2, 1, 0)
    return X

class NoFC_3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(NoFC_3, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.LeakyReLU())

    def forward(self, x):
        return self.layer(x)

# 参数形式定义
class NoHiT(nn.Module):
    def __init__(self, S1, S2, S3, r1, r2, r3):
        super(NoHiT, self).__init__()
        # self.W1 = nn.Parameter(torch.Tensor([0.5]))
        self.Z1 = nn.Parameter(init.xavier_normal_(torch.Tensor(r1, S1, r2)))
        self.Z2 = nn.Parameter(init.xavier_normal_(torch.Tensor(r2, S2, r3)))
        self.Z3 = nn.Parameter(init.xavier_normal_(torch.Tensor(r3, S3, r1)))

        self.g = nn.Sequential(NoFC_3(S3, S3))

    def forward(self):
        X_out = tr_product(self.Z1, self.Z2, self.Z3)

        X_out = self.g(X_out.permute(2, 1, 0).unsqueeze(0))
        X_out = X_out.squeeze(0).permute(1, 2, 0)

        return X_out