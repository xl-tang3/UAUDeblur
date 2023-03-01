import torch
import torch.nn as nn
from .fcn import fcn
from .skip_model import skip

class Abund_DIP(nn.Module):
    def __init__(self, rank, pad, out_channel):
        super(Abund_DIP, self).__init__()
        self.net_dir = {}
        self.rank = rank
        for i in range(self.rank):
            self.net_dir[str(i)] = nn.Sequential(skip(1, 1, num_channels_down=[16, 32, 64, 128, 128, 128],
                   num_channels_up=[16, 32, 64, 128, 128, 128],
                   num_channels_skip=[0, 0, 4, 4, 4, 4],
                   filter_size_down=[7, 7, 5, 5, 3, 3], filter_size_up=[7, 7, 5, 5, 3, 3],
                   upsample_mode='bilinear', downsample_mode='avg',
                   need_sigmoid=True, pad=pad, act_fun='LeakyReLU'))

        self.end_member = nn.Sequential(fcn(self.rank, out_channel, num_hidden=[128, 256, 256, 128]))

    def forward(self, x):
        if self.rank==1:
            y = self.net_dir['0'](x)
            return self.end_member(y.squeeze(1).permute(1, 2, 0))
        else:
            y = self.net_dir['0'](x)
            for i in range(1, self.rank):
                y = torch.cat((y, self.net_dir[str(i)](x)), 0)
            return self.end_member(y.squeeze(1).permute(1, 2, 0))
