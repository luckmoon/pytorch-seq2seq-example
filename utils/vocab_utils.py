#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from data.constants import PAD, UNK


class VocabHelper:
    def __init__(self,
                 vocab_file,
                 pad_token="<PAD>",
                 unk_token="<UNK>",
                 pad_idx=PAD,
                 unk_idx=UNK):
        assert pad_idx != unk_idx, "pad_idx cannot be equal with unk_idx"
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.vocab_file = vocab_file
        self.token_idx_dict = self.load_vocabs()
        self.idx_token_dict = self.get_idx_to_token_dict()

    def load_vocabs(self):
        with open(self.vocab_file, 'r') as fin:
            vocab_dict = json.load(fin)
        return vocab_dict

    def get_idx_to_token_dict(self):
        return {idx: token for token, idx in self.token_idx_dict.items()}

    def token2idx(self, token):
        return self.token_idx_dict.get(token, self.unk_idx)

    def idx2token(self, idx):
        return self.idx_token_dict.get(idx, self.unk_token)

    @property
    def vocab_size(self):
        return len(self.idx_token_dict)

    # def _load_pretrain(self):
    #     pretrain_dict = {}
    #     with open(os.path.join(args.pretrain_model_dir, "word2vec.mdl"), "r") as fin:
    #         for line in tqdm(fin.readlines()):
    #             word, vec = line.strip().split(" ", 1)
    #             vec = [float(x) for x in vec.split(" ")]
    #             pretrain_dict[word] = np.array(vec)
    #     return pretrain_dict
    #
    # def create_embedding_with_pretrain(self):
    #     """
    #     不在pretrain词表中的随机初始化
    #     :return:
    #     """
    #     np.random.seed(args.seed)
    #     pretrain_dict = self._load_pretrain()
    #     word_embeddings = np.random.rand(self.vocab_size, args.embedding_size)
    #     hit_pretrain_cnt = 0
    #     for idx, word in self.idx_token_dict.items():
    #         if word in pretrain_dict:
    #             hit_pretrain_cnt += 1
    #             word_embeddings[idx] = pretrain_dict[word]
    #     hit_rate = hit_pretrain_cnt / self.vocab_size
    #     word_embeddings[self.pad_idx] = np.zeros(args.embedding_size, dtype=np.float32)
    #     logger.info(f"create embedding from pretrained lm hit {hit_pretrain_cnt}/{self.vocab_size}={hit_rate:.4f}")
    #     return word_embeddings
