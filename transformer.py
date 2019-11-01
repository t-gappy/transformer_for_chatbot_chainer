"""Reference:

chainer implementation of Transformer
    https://github.com/soskek/attention_is_all_you_need
Inverse Token Frequency Loss
    https://arxiv.org/abs/1811.08100
Subword Regularization
    https://www.anlp.jp/proceedings/annual_meeting/2018/pdf_dir/B1-5.pdf
"""
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from layers import *
from tokenizer import Tokenizer
import copy

class TransformerConfig(object):
    def __init__(self,
                 tokenizer_dir = "dataset/trained_model.model",
                 dict_dir = "dataset/trained_model.vocab",
                 freq_dir = "dataset/piece_frequency.txt",
                 vocab_size = 32000,
                 layer_num = 6,
                 unit_num = 512,
                 parallel_num = 8,
                 max_length = 100,
                 dropout_rate = 0.1,
                 bos_id = 1,
                 eos_id = 2,
                 pad_id = 3,
                 label_smoothing = True,
                 smooth_eps = 0.1,
                 itf_lambda = 0.3,
                 augmentation = False,
                 ):
        self.tokenizer_dir = tokenizer_dir
        self.dict_dir = dict_dir
        self.freq_dir = freq_dir
        self.vocab_size = vocab_size
        self.layer_num = layer_num
        self.unit_num = unit_num
        self.parallel_num = parallel_num
        self.parallel_unit = unit_num / parallel_num
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing
        self.smooth_eps = smooth_eps
        self.itf_lambda = itf_lambda
        self.augmentation = augmentation

class Transformer(chainer.Chain):
    def __init__(self, config):
        self.config = config
        self.label_smoothing = config.label_smoothing
        self.position_encoding = self._init_position_encoding(
                    config.max_length, config.unit_num)
        self.tokenizer = Tokenizer(config.tokenizer_dir, config.dict_dir, config.augmentation)

        frequency = []
        with open(config.freq_dir) as f:
            for line in f:
                line = line.rstrip()
                frequency.append(line)
        self.itf = 1 / (np.array(frequency, dtype=np.float32)+1)**config.itf_lambda

        super(Transformer, self).__init__()
        with self.init_scope():
            self.source_embed = L.EmbedID(
                config.vocab_size, config.unit_num, ignore_label=config.pad_id)
            self.enc = Encoder(config)
            self.target_embed = L.EmbedID(
                config.vocab_size, config.unit_num, ignore_label=config.pad_id)
            self.dec = Decoder(config)


    def forward(self, x_s, x_t, translate=False):
        """
            args
                x_s: array of padded source sentences.
                x_t: array of padded target sentences.
                translate: whether this function used for translate or not.
            returns
                dec_out: encoder-decoder model's output.
                enc_out: encoder's output used for translation.
        """
        length_s, length_t = x_s.shape[1], x_t.shape[1]
        h_s = self.source_embed(x_s)
        h_t = self.target_embed(x_t)
        h_s += self.xp.array(self.position_encoding[None, :length_s])
        h_t += self.xp.array(self.position_encoding[None, :length_t])
        h_s = F.transpose(h_s, (0, 2, 1))
        h_t = F.transpose(h_t, (0, 2, 1))

        src_self_mask = self._get_padding_mask(x_s, x_s, self.config.pad_id)
        tgt_self_mask = self._get_padding_mask(x_t, x_t, self.config.pad_id)
        tgt_future_mask = self._get_future_mask(x_t)
        tgt_self_mask *= tgt_future_mask
        src_tgt_mask = self._get_padding_mask(x_s, x_t, self.config.pad_id)

        enc_out = self.enc(h_s, src_self_mask)
        dec_out = self.dec(h_t, enc_out, tgt_self_mask, src_tgt_mask)

        B, D, L = dec_out.shape
        dec_out = F.transpose(dec_out, (0, 2, 1)).reshape(B*L, D)
        dec_out = F.linear(dec_out, self.target_embed.W)

        if translate:
            return dec_out, enc_out
        else:
            return dec_out

    def __call__(self, x_s, x_t):
        """
            args
                x_s: list of source sentences
                    ["こんにちは", "あああああ", ...]
                x_t: list of target sentence
                    ["こんにちは", "アババババ", ...]
            returns
                loss: calculated loss (Variable)
        """
        x_s = self.tokenizer.tokenize_sentences(x_s)
        x_t = self.tokenizer.tokenize_sentences(x_t)
        x_s = self._get_padded_sentence(x_s, pad_id=self.config.pad_id)
        x_t = self._get_padded_sentence(x_t, pad_id=self.config.pad_id, eos_id=self.config.eos_id)

        batch_t, length_t = x_t.shape
        y_t = copy.deepcopy(x_t).reshape((batch_t*length_t))
        bos_ids = self.xp.repeat(self.xp.array(
                [self.config.bos_id], dtype=np.int32), batch_t, axis=0)[..., None]
        x_t = self.xp.concatenate([bos_ids, x_t[:, :length_t-1]], axis=1)

        y_pred = self.forward(x_s, x_t)

        if self.label_smoothing:
            loss = self._label_smoothed_sce(y_pred, y_t, eps=self.config.smooth_eps,
                itf=self.itf, ignore_label=self.config.pad_id)
        else:
            loss = F.softmax_cross_entropy(y_pred, y_t, ignore_label=self.config.pad_id)

        accuracy = F.accuracy(y_pred, y_t, ignore_label=self.config.pad_id)
        perplexity = self.xp.exp(loss.data)
        # print("loss: {}, perp: {}, acc: {}".format(loss.data, perplexity, accuracy.data))
        chainer.report({"loss": loss.data,
                        "perp": perplexity,
                        "acc": accuracy.data}, self)
        return loss

    def translate(self, x_s, max_length=65, beam=None):
        """
            args
                x_s: list of source sentences.
                    ["こんにちは", "あああああ", ...]
                max_length: max times of auto-regression
                beam: beam breadth in beam-search
                    '0' or 'None' means 'don't use beam-search'.
            returns
                translated: list of inferenced sentence(type:String) list.
        """
        batch_size = len(x_s)
        x_s = self.tokenizer.tokenize_sentences(x_s)
        x_s = self._get_padded_sentence(x_s, self.config.pad_id)
        x_t = self.xp.array([self.config.bos_id]*batch_size,
                            dtype=np.int32).reshape(batch_size, 1)
        eos_flags = self.xp.zeros((batch_size, 1), dtype=np.int32)
        y_pred, enc_out = self.forward(x_s, x_t, translate=True)

        with chainer.no_backprop_mode():
            with chainer.using_config("train", False):
                if beam:
                    # first search
                    # x_t, x_s shape: (batch, length) -> (batch*beam, length)
                    x_t = self.xp.concatenate(
                        [x_t[:, None, :]]*beam, axis=1).reshape(beam*batch_size, 1)
                    x_s = self.xp.concatenate(
                        [x_s[:, None, :]]*beam, axis=1).reshape(beam*batch_size, x_s.shape[1])
                    scores = self.xp.zeros((batch_size*beam), dtype=np.float32)
                    candidates, s = self._get_beam_results(y_pred.data, beam, 1)
                    scores += s
                    x_t = self.xp.concatenate([x_t, candidates[..., None]], axis=1)

                    x_t = self._beam_translate(max_length-2, x_s, x_t, None, scores, max_length, beam)

                else:
                    x_t = self.xp.concatenate([x_t, self.xp.argmax(y_pred.data, axis=1)[..., None]], axis=1)

                    for i in range(max_length-1):
                        y_pred = self._translate_forward(enc_out, x_s, x_t)
                        #print(i, self.xp.mean(y_pred.data), self.xp.max(y_pred.data), self.xp.min(y_pred.data))
                        y_inds = self.xp.argmax(y_pred.data, axis=1)[i+1::i+2, None]
                        x_t = self.xp.concatenate([x_t, y_inds], axis=1)
                        eos_flags += (y_inds == self.config.eos_id)
                        if self.xp.all(eos_flags):
                            break

        translated = [[] for i in range(batch_size)]
        for b, sentence in enumerate(x_t[:, 1:]):
            for w in sentence:
                if w == self.config.eos_id:
                    break
                translated[b].append(w)

        translated = self.tokenizer.detokenize_sentences(translated)
        return translated

    def _beam_translate(self, depth, x_s, x_t, enc_out, scores, max_length, beam):
        """recurrent beam search for translate.
            args
                depth: controll inferencing depth.
                    (this function perform recurrently)
                x_s: array of source sentences. (batch*beam, length)
                    Note this x_s is not the same as arg of 'translate' function.
                x_t: array of target sentences. (batch*beam, length)
                    this arg changes gradually in auto-regression.
                enc_out: encoder's output (fixed after calculated once)
                scores: candidates scores for selecting good output.
                max_length: max times of auto-regression.
                beam: beam breadth in beam-search.
            returns
                x_t: predicted (intermediate) sentence.
        """
        batch_size = len(x_t)
        if depth == max_length-2:
            # y_pred shapes (batch*beam*2, vocab_size), and get candidates from y_pred
            y_pred, enc_out = self.forward(x_s, x_t, translate=True)
        else:
            y_pred = self._translate_forward(enc_out, x_s, x_t)

        candidates, s = self._get_beam_results(y_pred.data, beam, max_length-depth)

        # x_t shape -> (batch*beam*beam, L) -> (batch, beam*beam, L)
        x_t = self.xp.concatenate([x_t[:, None, :]]*beam, axis=1)
        x_t = x_t.reshape(beam*batch_size, max_length-depth)
        x_t = self.xp.concatenate([x_t, candidates[..., None]], axis=1)
        x_t = x_t.reshape(batch_size//beam, beam*beam, max_length-depth+1)

        # score the same as x_t
        scores = self.xp.concatenate([scores[:, None]]*beam, axis=1)
        scores = scores.reshape(beam*batch_size, )
        scores += s
        scores = scores.reshape(batch_size//beam, beam*beam)

        if depth == 0:
            best_sentence_ind = self.xp.argmax(scores, axis=1)
            x_t = x_t[self.xp.arange(batch_size//beam), best_sentence_ind]
            return x_t

        # sorting by scores, getting sentence-candidates for next depth.
        beam_indeces = self.xp.argsort(scores, axis=1)[:, ::-1][:, :beam]
        beam_indeces = self.xp.concatenate(beam_indeces, axis=0)
        batch_indeces = self.xp.arange(batch_size//beam)
        batch_indeces = self.xp.concatenate([batch_indeces[..., None]]*beam, axis=1)
        batch_indeces = batch_indeces.reshape(batch_size, )
        x_t = x_t[batch_indeces, beam_indeces]

        scores = self.xp.sort(scores, axis=1)[:, ::-1][:, :beam]
        scores = self.xp.concatenate(scores, axis=0)

        if self.xp.all(self.xp.any(x_t == 2, axis=1)):
            scores = scores.reshape(batch_size//beam, beam)
            best_sentence_ind = self.xp.argmax(scores, axis=1)
            x_t = x_t.reshape(batch_size//beam, beam, x_t.shape[1])
            x_t = x_t[self.xp.arange(batch_size//beam), best_sentence_ind]
            return x_t

        x_t = self._beam_translate(depth-1, x_s, x_t, enc_out, scores, max_length, beam)

        return x_t

    def _get_beam_results(self, y_pred, beam, position):
        """beam results should be (batch*beam, length).
            args
                y_pred: decoder's output in auto-regression.
                beam: beam size of candidate getting
                position: specify where candidates should be get from.
                    if position is 2, <> position below will be candidates.
                    [<batch_0>, batch_1, <batch_2>, batch_3, ..., <batch_2n>]
            returns
                candidates: top beam-th candidates on y_pred.
                scores: top beam-th scores on y_pred.
        """
        candidates = self.xp.argsort(y_pred)[:, ::-1][position-1::position, :beam]
        candidates = self.xp.concatenate(candidates, axis=0)
        scores = self.xp.sort(y_pred)[:, ::-1][position-1::position, :beam]
        scores = self.xp.concatenate(scores, axis=0)

        return candidates, scores

    def _translate_forward(self, enc_out, x_s, x_t):
        """reusing enc_out for efficient calculation.
            args
                enc_out: encoder's output (fixed after calculated once)
                x_s: array of source sentences.
                    Note this x_s is not the same as arg of 'translate' function.
                x_t: array of target sentences.
                    this arg changes gradually in auto-regression.
            returns
                dec_out: decoder's output
        """
        length_t = x_t.shape[1]
        h_t = self.target_embed(x_t)
        h_t += self.position_encoding[None, :length_t]
        h_t = F.transpose(h_t, (0, 2, 1))

        tgt_self_mask = self._get_padding_mask(x_t, x_t, self.config.pad_id)
        tgt_future_mask = self._get_future_mask(x_t)
        tgt_self_mask *= tgt_future_mask
        src_tgt_mask = self._get_padding_mask(x_s, x_t, self.config.pad_id)

        dec_out = self.dec(h_t, enc_out, tgt_self_mask, src_tgt_mask)

        B, D, L = dec_out.shape
        dec_out = F.transpose(dec_out, (0, 2, 1)).reshape(B*L, D)
        dec_out = F.linear(dec_out, self.target_embed.W)

        return dec_out

    def _init_position_encoding(self, max_length, unit_num):
        half_dim = unit_num // 2
        dim_positions = - (np.arange(half_dim) * 2 / unit_num)
        dim_positions = 10000 ** dim_positions

        word_positions = np.arange(max_length)
        general_encode = word_positions[..., None] * dim_positions[None, ...]
        even_dims = np.sin(general_encode)
        odd_dims = np.cos(general_encode)

        position_encoding = np.concatenate([even_dims[..., None], odd_dims[..., None]], axis=2)
        position_encoding = position_encoding.reshape(max_length, unit_num)

        return position_encoding.astype(np.float32)

    def _get_padded_sentence(self, xs, pad_id, eos_id=None):
        batch_size = len(xs)
        max_length = max([len(x) for x in xs])

        if eos_id:
            padded_sentence = self.xp.full((batch_size, max_length+2), pad_id, dtype=np.int32)
            for i, x in enumerate(xs):
                x_eos = x + [eos_id]
                padded_sentence[i, :len(x_eos)] = self.xp.array(x_eos, dtype=np.int32)
        else:
            padded_sentence = self.xp.full((batch_size, max_length), pad_id, dtype=np.int32)
            for i, x in enumerate(xs):
                padded_sentence[i, :len(x)] = self.xp.array(x, dtype=np.int32)

        return padded_sentence

    def _get_padding_mask(self, key, query, pad_id):
        """
            args
                key: key in attention.
                    in source-target attention, this means 'source'
                    shape is (batch, length).
                query: query in attention.
                    in source-target attention, this means 'target'
                    shape is (batch, length).
            returns
                mask: (batch, q-length, k-length) shape xp-array.
        """
        query_mask = query != pad_id
        key_mask = key != pad_id
        mask = key_mask[:, None, :] * query_mask[..., None]
        return mask

    def _get_future_mask(self, x):
        """
            args
                x: target's input array
                    shape is (batch, length)
            returns
                mask: mask for future-ignoring.
                    when batch is 1 and length is 4,
                    [[[ True, False, False, False],
                      [ True,  True, False, False],
                      [ True,  True,  True, False],
                      [ True,  True,  True,  True]]]
                    will be return.
        """
        batch, length = x.shape
        arange = self.xp.arange(length)
        future_mask = (arange[None, ] <= arange[:, None])[None, ...]
        future_mask = self.xp.concatenate([future_mask]*batch, axis=0)
        return future_mask

    def _label_smoothed_sce(self, y, t, eps, itf, ignore_label=None):
        """note: variable 'batch_size' means batch*length of the task.
            args
                y: model output (batch*length, vocab_size)
                t: ground truth (batch*length, )
                    this value is index of truth word in vocab.
                eps: epsilon for label-smoothing.
                itf: array of inverse token frequency.
                ignore_label: word whitch should be ignored for calculation.
            returns
                loss: loss (Variable) between y and label-smoothed-t.
        """
        xp = chainer.cuda.get_array_module(t)
        batch_size, vocab_size = y.shape
        func_u = eps / vocab_size

        smoothed_t = xp.zeros_like(y.data).astype(np.float32)
        smoothed_t[xp.arange(batch_size), t] = 1 - eps# + func_u
        smoothed_t += func_u

        loss = F.log_softmax(y) * smoothed_t
        normalizer = batch_size
        if ignore_label:
            ignore_mask = t != ignore_label
            normalizer = xp.sum(ignore_mask)
            loss = ignore_mask[..., None] * loss

        loss = loss * self.xp.array(itf[None, ...], dtype=np.float32)
        loss = - F.sum(loss) / normalizer

        return loss
