import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import sys
sys.path.append("../")
from layers import *

class Encoder(chainer.ChainList):
    def __init__(self, config):
        super(Encoder, self).__init__()
        for layer in range(config.layer_num):
            self.add_link(EncoderBlock(config))

    def __call__(self, x, mask):
        for layer in self.children():
            x = layer(x, mask)
        return x


class EncoderBlock(chainer.Chain):
    def __init__(self, config):
        self.dropout_rate = config.dropout_rate
        super(EncoderBlock, self).__init__()
        with self.init_scope():
            self.mha = MultiHeadAttention(config, self_attention=True)
            self.ln1 = LayerNormalization3D(config)
            self.ffl = FeedForwardLayer(config)
            self.ln2 = LayerNormalization3D(config)

    def __call__(self, x, mask):
        h1 = self.ln1(x)
        h1 = self.mha(h1, mask=mask)
        h1 = x + F.dropout(h1, self.dropout_rate)

        h2 = self.ln2(h1)
        h2 = self.ffl(h2)
        y = h1 + F.dropout(h2, self.dropout_rate)
        
        return y
