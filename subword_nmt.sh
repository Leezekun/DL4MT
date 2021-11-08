subword-nmt learn-joint-bpe-and-vocab --input data/train_clean.en data/train_clean.ha -s 10000 -o data/codes.en-ha --write-vocabulary data/vocab.en data/vocab.ha

subword-nmt apply-bpe -c data/codes.en-ha --vocabulary data/vocab.en --vocabulary-threshold 50 < data/train_clean.en > data/train.BPE.en
subword-nmt apply-bpe -c data/codes.en-ha --vocabulary data/vocab.ha --vocabulary-threshold 50 < data/train_clean.ha > data/train.BPE.ha

python build_vocabulary.py data/train.BPE.en data/train.BPE.ha

subword-nmt apply-bpe -c data/codes.en-ha --vocabulary data/vocab.en --vocabulary-threshold 50 < data/dev.en > data/dev.BPE.en
subword-nmt apply-bpe -c data/codes.en-ha --vocabulary data/vocab.ha --vocabulary-threshold 50 < data/dev.ha > data/dev.BPE.ha
