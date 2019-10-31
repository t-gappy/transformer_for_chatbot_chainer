import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import sys
sys.path.append("../")
from layers import *

class Decoder(chainer.ChainList):
    def __init__(self, config):
        super(Decoder, self).__init__()
        for layer in range(config.layer_num):
            self.add_link(DecoderBlock(config))

    def __call__(self, x, enc_out, self_mask, st_mask):
        for layer in self.children():
            x = layer(x, enc_out, self_mask, st_mask)
        return x


class DecoderBlock(chainer.Chain):
    def __init__(self, config):
        self.dropout_rate = config.dropout_rate
        super(DecoderBlock, self).__init__()
        with self.init_scope():
            self.self_mha = MultiHeadAttention(config, self_attention=True)
            self.ln1 = LayerNormalization3D(config)
            self.st_mha = MultiHeadAttention(config, self_attention=False)
            self.ln2_dec = LayerNormalization3D(config)
            self.ln2_enc = LayerNormalization3D(config)
            self.ffl = FeedForwardLayer(config)
            self.ln3 = LayerNormalization3D(config)

    def __call__(self, x, enc_out, self_mask, st_mask):
        h1 = self.ln1(x)
        h1 = self.self_mha(h1, mask=self_mask)
        h1 = x + F.dropout(h1, self.dropout_rate)

        h2 = self.ln2_dec(h1)
        enc_out = self.ln2_enc(enc_out)
        h2 = self.st_mha(h2, enc_out=enc_out, mask=st_mask)
        h2 = h1 + F.dropout(h2, self.dropout_rate)

        h3 = self.ln3(h2)
        h3 = self.ffl(h3)
        y = h2 + F.dropout(h3, self.dropout_rate)

        return y
