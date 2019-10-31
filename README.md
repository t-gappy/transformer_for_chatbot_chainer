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

# You Must Prepare
- dataset
- training code
- folder "./dataset"
- ./dataset/trained_model.model
  - sentencepiece tokenizer's model file
- ./dataset/trained_model.vocab
  - sentencepiece tokenizer's vocab file
- ./dataset/piece_frequency.txt
  - token frequency file
  - one line contains the frequency of the token in the same line in .vocab file
    - if the token "hello" appear 300 times, "300" is in the "hello line".
    - if you set the "augmentation" arg to True in TransformerConfig class, you should take some step like "calculate expected value".
