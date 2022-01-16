#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import json
from collections import Counter

from tqdm import tqdm

from data.constants import PAD, BOS, EOS, UNK
from utils.utils import AttrDict


def load_sents(file_path):
    sents = []
    with codecs.open(file_path) as file:
        for sent in tqdm(file.readlines()):
            tokens = [token for token in sent.split()]
            sents.append(tokens)
    return sents


# input:sents:由1个file得到list。
#     output:
def build_counter(sents):
    counter = Counter()
    for sent in tqdm(sents):
        counter.update(sent)
    return counter


def build_vocab(counter, max_vocab_size):
    token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS, '<UNK>': UNK}
    token2id.update(
        {token: _id + 4 for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size)))})
    return token2id


def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
    seq = []
    if append_BOS:
        seq.append(BOS)
    seq.extend([token2id.get(token, UNK) for token in tokens])
    if append_EOS:
        seq.append(EOS)
    return seq


def main():
    src_file_path = "../jfleg/dev/dev.src"
    tgt_file_path = "../jfleg/dev/dev.ref1"
    src_sents = load_sents(src_file_path)
    tgt_sents = load_sents(tgt_file_path)
    src_counter = build_counter(src_sents)
    tgt_counter = build_counter(tgt_sents)
    if share_vocab:
        src_vocab = build_vocab(src_counter + tgt_counter, max_vocab_size)
        tgt_vocab = src_vocab
    else:
        src_vocab = build_vocab(src_counter, max_vocab_size)
        tgt_vocab = build_vocab(src_counter, max_vocab_size)

    with open(src_vocab_file, 'w') as fout:
        json.dump(src_vocab, fout, ensure_ascii=False, indent=2)
    with open(tgt_vocab_file, 'w') as fout:
        json.dump(tgt_vocab, fout, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    share_vocab = True
    max_vocab_size = 10000

    vocab_dir = "./dataset"
    if not os.path.isdir(vocab_dir):
        os.makedirs(vocab_dir)
    src_vocab_file = os.path.join(vocab_dir, "src_vocab.json")
    tgt_vocab_file = os.path.join(vocab_dir, "tgt_vocab.json")

    main()
