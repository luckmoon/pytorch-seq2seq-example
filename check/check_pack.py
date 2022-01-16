#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

emb = torch.tensor([[7, 8, 3, 5, 6, 7, 7], [1, 2, 3, 0, 0, 0, 0]]).transpose(0, 1)
src_lens = [7, 5]
packed_emb = nn.utils.rnn.pack_padded_sequence(emb, src_lens, batch_first=False)
print(packed_emb)
