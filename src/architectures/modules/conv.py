import torch.nn as nn

class SpatialPreservedConv(nn.Conv2d):
    """
    To keep spatial size of input same as output, this code only work for stride step = 1, and kernel size is odd number.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        if kernel_size % 2 == 0:
            NotImplementedError("When stride is 1, this only works for odd kernel size.")

        super(SpatialPreservedConv, self).__init__(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)