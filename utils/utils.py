#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess

import torch


class AttrDict(dict):
    """
    Access dictionary keys like attribute
    https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """

    def __init__(self, *av, **kav):  # 【que】
        dict.__init__(self, *av, **kav)
        self.__dict__ = self  # [que]


def sequence_mask(sequence_length, max_len=None):
    """
    Caution: Input and Return are VARIABLE.
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand

    return mask


def variable2numpy(var):
    """ For tensorboard visualization """
    return var.data.cpu().numpy()


def get_gpu_memory_usage(device_id):
    """Get the current gpu usage. """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map[device_id]


def compute_grad_norm(parameters, norm_type=2):
    """ Ref: http://pytorch.org/docs/0.3.0/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm


def load_checkpoint(checkpoint_path):
    # It's weird that if `map_location` is not given, it will be extremely slow.
    return torch.load(checkpoint_path, map_location=lambda storage, loc: storage)


def save_checkpoint(checkpoint_dict, checkpoint_path):
    directory, filename = os.path.split(os.path.abspath(checkpoint_path))
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(checkpoint_dict, checkpoint_path)


def get_opts(load_ckpt):
    if load_ckpt:
        # Modify this path.
        checkpoint_path = './checkpoints/seq2seq_2018-02-07 20:30:47_acc_88.15_loss_12.85_step_135000.pt'
        checkpoint = load_checkpoint(checkpoint_path)
        opts = checkpoint['opts']
    else:
        opts = AttrDict()

        # Configure models
        opts.word_vec_size = 300
        opts.rnn_type = 'LSTM'
        opts.hidden_size = 512
        opts.num_layers = 2
        opts.dropout = 0.3
        opts.bidirectional = True
        opts.attention = True
        opts.share_embeddings = True
        opts.pretrained_embeddings = True
        opts.fixed_embeddings = True
        opts.tie_embeddings = True  # Tie decoder's input and output embeddings

        # Configure optimization
        opts.max_grad_norm = 2
        opts.learning_rate = 0.001
        opts.weight_decay = 1e-5  # L2 weight regularization

        # Configure training
        opts.max_seq_len = 100  # max sequence length to prevent OOM.
        opts.num_epochs = 1
        opts.print_every_step = 20
        opts.save_every_step = 5000
    return opts
