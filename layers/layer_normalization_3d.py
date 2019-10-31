import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class LayerNormalization3D(chainer.Chain):
    def __init__(self, config):
        super(LayerNormalization3D, self).__init__()
        with self.init_scope():
            self.ln = L.LayerNormalization(config.unit_num)

    def __call__(self, x):
        """
            args
                x: main features in the model
                   Variable in (batch, dim, length)
            returns
                y: output, same size with x
        """
        B, D, L = x.shape
        x = F.transpose(x, (0, 2, 1))
        x = F.reshape(x, (B*L, D))
        y = self.ln(x)
        y = F.reshape(y, (B, L, D))
        y = F.transpose(y, (0, 2, 1))
        return y
