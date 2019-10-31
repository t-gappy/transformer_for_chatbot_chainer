# transformer_for_chatbot
chainer implementation of transformer for chatbot.<br>
this implementation considers the diversity of model's ouput.

# Contents
- README.md
- transformer.py
- tokenizer.py
- utils.py
- layers
  - init.py
  - decoder.py
  - encoder.py
  - feed_forward_layer.py
  - layer_normalization_3d.py
  - multi_head_attention.py

# Must Prepare
- dataset
- training code
- folder "./dataset"
- ./dataset/trained_model.model
  - sentencepiece tokenizer's model file
- ./dataset/trained_model.vocab
  - sentencepiece tokenizer's vocab file
- ./dataset/piece_frequency.txt
  - token frequency file
  - one line contains the frequency of the token in the same line with .vocab file
    - if the token "hello" appears 300 times, "300" is in the "hello line".
    - if you set the "augmentation" arg to True in TransformerConfig class, you should take some steps like "calculate expected value".

# Reference
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - [chainer implementation](https://github.com/soskek/attention_is_all_you_need)
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
- [Another Diversity-Promoting Objective Function for Neural Dialogue Generation](https://arxiv.org/abs/1811.08100)
- [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://www.aclweb.org/anthology/P18-1007/)
