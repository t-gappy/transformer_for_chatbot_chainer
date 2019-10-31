import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import copy

class MultiHeadAttention(chainer.Chain):
    def __init__(self, config, self_attention=True):
        self.scale = 1. / config.parallel_unit ** 0.5
        self.self_attention = self_attention
        self.parallel_num = config.parallel_num
        self.dropout_rate = config.dropout_rate

        super(MultiHeadAttention, self).__init__()
        with self.init_scope():
            if self_attention:
                self.W = L.Convolution2D(config.unit_num, config.unit_num*3,
                                         ksize=1, stride=1, pad=0, nobias=True)
            else:
                self.W_KV = L.Convolution2D(config.unit_num, config.unit_num*2,
                                            ksize=1, stride=1, pad=0, nobias=True)
                self.W_Q = L.Convolution2D(config.unit_num, config.unit_num,
                                           ksize=1, stride=1, pad=0, nobias=True)
            self.linear = L.Convolution2D(config.unit_num, config.unit_num,
                                          ksize=1, stride=1, pad=0)

    def __call__(self, x, enc_out=None, mask=None):
        """
            args
                x: paralleled main features in the model
                   Variable in (batch, hidden_dim, length)
                u: hidden features from Encoder
                   Variable in (batch, hidden_dim, length)
                mask: padding-mask or future-mask
                   xp-array in (batch, length, length)
                   an element takes 'False' when pad/future, otherwise 'True'
            returns
        """
        # ksize-1-convolution results in parallel linear projections
        if self.self_attention:
            qkv = F.squeeze(self.W(F.expand_dims(x, axis=3)), axis=3)
            query, key, value = F.split_axis(qkv, 3, axis=1)
        else:
            query = F.squeeze(self.W_Q(F.expand_dims(x, axis=3)), axis=3)
            kv = F.squeeze(self.W_KV(F.expand_dims(enc_out, axis=3)), axis=3)
            key, value = F.split_axis(kv, 2, axis=1)

        # make q,k,v into (batch*parallel, dim/parallel, length)shape
        query = F.concat(F.split_axis(query, self.parallel_num, axis=1), axis=0)
        key = F.concat(F.split_axis(key, self.parallel_num, axis=1), axis=0)
        value = F.concat(F.split_axis(value, self.parallel_num, axis=1), axis=0)
        mask = self.xp.concatenate([mask]*self.parallel_num, axis=0)

        attention_weight = F.batch_matmul(query, key, transa=True) * self.scale
        attention_weight = F.where(mask, attention_weight,
            self.xp.full(attention_weight.shape, -np.inf, dtype=np.float32))
        attention_weight = F.softmax(attention_weight, axis=2)
        attention_weight = F.dropout(attention_weight, self.dropout_rate)
        attention_weight = F.where(self.xp.isnan(attention_weight.data),
            self.xp.full(attention_weight.shape, 0, dtype=np.float32), attention_weight)
        self.attention_weight = copy.deepcopy(attention_weight.data)

        # attention: (batch, q-length, k-length) -> (batch, 1, q-length, k-length)
        # value: (batch, dim/parallel, k-length) -> (batch, dim/parallel, 1, k-length)
        attention_weight, value = F.broadcast(attention_weight[:, None], value[:, :, None])
        weighted_sum = F.sum(attention_weight*value, axis=3)
        weighted_sum = F.concat(F.split_axis(weighted_sum, self.parallel_num, axis=0), axis=1)

        weighted_sum = F.squeeze(self.linear(F.expand_dims(weighted_sum, axis=3)), axis=3)
        return weighted_sum
