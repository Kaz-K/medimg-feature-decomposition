import torch.nn as nn
from torch.nn.utils import spectral_norm


def apply_spectral_norm(net):
    def _add_spectral_norm(m):
        classname = m.__class__.__name__
        print(classname)
        if classname.find('Conv2d') != -1:
            m = spectral_norm(m)
        elif classname.find('Linear') != -1:
            m = spectral_norm(m)

    print('applying normalization [spectral_norm]')
    net.apply(_add_spectral_norm)
