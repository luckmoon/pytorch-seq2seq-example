#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data.constants import BOS, EOS, UNK
from utils.vocab_utils import VocabHelper


class NMTDataset(Dataset):
    def __init__(self, src_path, tgt_path, share_vocab=True):
        """ Note: If src_vocab, tgt_vocab is not given, it will build both vocabs.
            Args:
            - src_path, tgt_path: text file with tokenized sentences.
        """
        print('=' * 100)
        print('Dataset preprocessing log:')

        print('- Loading and tokenizing source sentences...')
        self.src_sents = self.load_sents(src_path)
        print('- Loading and tokenizing target sentences...')
        self.tgt_sents = self.load_sents(tgt_path)

        self.src_vocab = VocabHelper(vocab_file="./dataset/src_vocab.json")
        self.tgt_vocab = VocabHelper(vocab_file="./dataset/tgt_vocab.json")

        print('=' * 100)
        print('Dataset Info:')
        print('- Number of source sentences: {}'.format(len(self.src_sents)))
        print('- Number of target sentences: {}'.format(len(self.tgt_sents)))
        print('- Source vocabulary size: {}'.format(self.src_vocab.vocab_size))
        print('- Target vocabulary size: {}'.format(self.tgt_vocab.vocab_size))
        print('- Shared vocabulary: {}'.format(share_vocab))
        print('=' * 100 + '\n')

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        tgt_sent = self.tgt_sents[index]
        src_seq = self.tokens2ids(src_sent, self.src_vocab.token_idx_dict, append_BOS=False, append_EOS=True)
        tgt_seq = self.tokens2ids(tgt_sent, self.tgt_vocab.token_idx_dict, append_BOS=False, append_EOS=True)

        return src_sent, tgt_sent, src_seq, tgt_seq

    def load_sents(self, file_path):
        sents = []
        with codecs.open(file_path) as file:
            for sent in tqdm(file.readlines()):
                tokens = [token for token in sent.split()]
                sents.append(tokens)
        return sents

    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        if append_BOS:
            seq.append(BOS)
        seq.extend([token2id.get(token, UNK) for token in tokens])
        if append_EOS:
            seq.append(EOS)
        return seq


def collate_fn(data):
    """
    Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
        - src_sents, tgt_sents: batch of original tokenized sentences
        - src_seqs, tgt_seqs: batch of original tokenized sentence ids
    Returns:
        - src_sents, tgt_sents (tuple): batch of original tokenized sentences
        - src_seqs, tgt_seqs (variable): (max_src_len, batch_size)
        - src_lens, tgt_lens (tensor): (batch_size)

    """

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
    # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
    # *source* sequence, which is in the EncoderRNN
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences.
    src_sents, tgt_sents, src_seqs, tgt_seqs = zip(*data)

    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lens = _pad_sequences(src_seqs)
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seqs)

    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0, 1)
    tgt_seqs = tgt_seqs.transpose(0, 1)

    return src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens
