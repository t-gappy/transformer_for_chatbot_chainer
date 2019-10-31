import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

class FeedForwardLayer(chainer.Chain):
    def __init__(self, config):
        super(FeedForwardLayer, self).__init__()
        with self.init_scope():
            self.linear1 = L.Convolution2D(config.unit_num, config.unit_num*4,
                                           ksize=1, stride=1, pad=0)
            self.linear2 = L.Convolution2D(config.unit_num*4, config.unit_num,
                                           ksize=1, stride=1, pad=0)

    def __call__(self, x):
        """
            args
                x: main features in the model
                   Variable in (batch, dim, length)
            returns
                y: output, same size with x
        """
        x = F.expand_dims(x, axis=3)
        h = F.relu(self.linear1(x))
        y = F.squeeze(self.linear2(h), axis=3)
        return y
