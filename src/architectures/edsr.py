import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

from .modules.conv import SpatialPreservedConv

class EDSR(nn.Module):
    # Based on their modified version from: https://github.com/yinboc/liif

    url = {
        'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
        'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
        'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
        'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
        'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
        'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
    }

    # n_resblocks: 16, n_feats: 64, res_scale: 1, kernel_size: 3, scale: 2, rgb_range: 1, n_colors: 3, no_upsampling: false

    def __init__(self, *args, **kwargs):
        super(EDSR, self).__init__()
        self.hparams = OmegaConf.create(kwargs)
        
        n_resblocks = self.hparams.n_resblocks
        n_feats = self.hparams.n_feats

        # url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        # if url_name in url:
        #     self.url = url[url_name]
        # else:
        #     self.url = None

        kernel_size = self.hparams.kernel_size

        # define head module
        m_head = [SpatialPreservedConv(self.hparams.n_colors, n_feats, kernel_size)]

        # define body module
        # ResBlock(n_feats, kernel_size, act=nn.ReLU(True), res_scale=hparams.res_scale)
        m_body = [
            hydra.utils.instantiate(self.hparams.resblock, act=nn.ReLU(True)) for _ in range(n_resblocks)
        ]
        m_body.append(SpatialPreservedConv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        print(f"{self.hparams.upsampler}")
        if self.hparams.upsampler:
            self.out_dim = self.hparams.n_colors
            # define tail module
            m_tail = [
                hydra.utils.instantiate(self.hparams.upsampler),
                SpatialPreservedConv(n_feats, self.hparams.n_colors, kernel_size)
            ]
            self.tail = nn.Sequential(*m_tail)
        else:
            self.out_dim = n_feats

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        if self.hparams.upsampler:
            x = self.tail(res)
        else:
            x = res
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))