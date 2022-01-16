#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


USE_CUDA = torch.cuda.is_available()

device = get_device()

PAD = 0
BOS = 1
EOS = 2
UNK = 3
# {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS, '<UNK>': UNK}