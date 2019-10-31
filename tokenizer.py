import sentencepiece as spm

class Tokenizer(object):
    def __init__(self, model_dir, dict_dir, augmentation):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_dir)
        self.augmentation = augmentation

        self.piece2num = {}
        self.num2piece = {}
        with open(dict_dir) as fr:
            for i, line in enumerate(fr):
                line = line.rstrip().split("\t")
                self.piece2num[line[0]] = i
                self.num2piece[i] = line[0]

    def tokenize_sentences(self, sentence_list):
        output = []
        for sentence in sentence_list:
            if self.augmentation:
                pieces = self.sp.SampleEncodeAsPieces(sentence, 5, 0.1)
            else:
                pieces = self.sp.EncodeAsPieces(sentence)
            num_pieces = []
            for p in pieces:
                try:
                    num_pieces.append(self.piece2num[p])
                except KeyError:
                    num_pieces.append(self.piece2num["<unk>"])
            output.append(num_pieces)

        return output

    def detokenize_sentences(self, ids_list):
        sentence_list = []
        for ids in ids_list:
            sentence = []
            for id in ids:
                sentence.append(self.num2piece[id])
            sentence = "".join(sentence).replace("_", "")
            sentence_list.append(sentence)
        
        return sentence_list
