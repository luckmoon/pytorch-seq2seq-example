#!/usr/bin/env python
# coding: utf-8

# ## Batched Seq2Seq Example
# Based on the [`seq2seq-translation-batched.ipynb`](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb) from *practical-pytorch*, but more extra features.
# 
# This example runs grammatical error correction task where the source sequence is a grammatically erroneuous English sentence and the target sequence is an grammatically correct English sentence. The corpus and evaluation script can be download at: https://github.com/keisks/jfleg.
# 
# ### Extra features
# - Cleaner codebase
# - Very detailed comments for learners
# - Implement Pytorch native dataset and dataloader for batching
# - Correctly handle the hidden state from bidirectional encoder and past to the decoder as initial hidden state.
# - Fully batched attention mechanism computation (only implement `general attention` but it's sufficient). Note: The original code still uses for-loop to compute, which is very slow.
# - Support LSTM instead of only GRU
# - Shared embeddings (encoder's input embedding and decoder's input embedding)
# - Pretrained Glove embedding
# - Fixed embedding
# - Tie embeddings (decoder's input embedding and decoder's output embedding)
# - Tensorboard visualization
# - Load and save checkpoint
# - Replace unknown words by selecting the source token with the highest attention score. (Translation)
# 
# ### Cons
# Comparing to the state-of-the-art seq2seq library, OpenNMT-py, there are some stuffs that aren't optimized in this codebase:
# - Use CuDNN when possible (always on encoder, on decoder when input_feed 0)
# - Always avoid indexing / loops and use torch primitives.
# - When possible, batch softmax operations across time. ( this is the second complicated part of the code)
# - Batch inference and beam search for translation (this is the most complicated part of the code)
# 
# Thanks to the author of OpenNMT-py @srush for answering the questions for me! See https://github.com/OpenNMT/OpenNMT-py/issues/552

# In[1]:


# 第1遍：
# 第2遍：
# 第3遍：

import codecs
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import spacy

from data.constants import USE_CUDA, PAD, BOS, EOS, UNK, device
from data.data import NMTDataset, collate_fn
from model.model import EncoderRNN, LuongAttnDecoderRNN
from utils.utils import sequence_mask, variable2numpy, get_gpu_memory_usage, compute_grad_norm, \
    save_checkpoint, get_opts
from utils.vocab_utils import VocabHelper

""" Please download from here: 
1. Install spacy: https://spacy.io/usage/
2. Install model: https://spacy.io/usage/models
Recommend to install spacy since it is a very powerful NLP tool
"""


src_vocab_helper = VocabHelper("./dataset/src_vocab.json")
tgt_vocab_helper = VocabHelper("./dataset/tgt_vocab.json")
# src_vocab_size = src_vocab_helper.vocab_size

nlp = spacy.load('en_core_web_lg')  # For the glove embeddings

""" Enable GPU training """
print('Use_CUDA={}'.format(USE_CUDA))
if USE_CUDA:
    # You can change device by `torch.cuda.set_device(device_id)`
    print('current_device={}'.format(torch.cuda.current_device()))


def load_spacy_glove_embedding(spacy_nlp, vocab):
    vocab_size = vocab.vocab_size
    word_vec_size = spacy_nlp.vocab.vectors_length
    embedding = np.zeros((vocab_size, word_vec_size))
    unk_count = 0

    print('=' * 100)
    print('Loading spacy glove embedding:')
    print('- Vocabulary size: {}'.format(vocab_size))
    print('- Word vector size: {}'.format(word_vec_size))

    for token, index in tqdm(vocab.token_idx_dict.items()):
        if token == vocab.idx_token_dict[PAD]:
            continue
        elif token in [vocab.idx_token_dict[BOS], vocab.idx_token_dict[EOS], vocab.idx_token_dict[UNK]]:
            vector = np.random.rand(word_vec_size, )
        elif spacy_nlp.vocab[token].has_vector:
            vector = spacy_nlp.vocab[token].vector
        else:
            vector = embedding[UNK]
            unk_count += 1

        embedding[index] = vector

    print('- Unknown word count: {}'.format(unk_count))
    print('=' * 100 + '\n')

    return torch.from_numpy(embedding).float()


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
        
    The code is same as:
    
    weight = torch.ones(tgt_vocab_size)
    weight[padding_idx] = 0
    criterion = nn.CrossEntropyLoss(weight.cuda(), size_average)
    loss = criterion(logits_flat, losses_flat)
    """
    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    # Note: mask need to bed casted to float!
    losses = losses * mask.float()
    loss = losses.sum() / mask.float().sum()

    # (batch_size * max_tgt_len,)
    pred_flat = log_probs_flat.max(1)[1]
    # (batch_size * max_tgt_len,) => (batch_size, max_tgt_len) => (max_tgt_len, batch_size)
    pred_seqs = pred_flat.view(*target.size()).transpose(0, 1).contiguous()
    # (batch_size, max_len) => (batch_size * max_tgt_len,)
    mask_flat = mask.view(-1)

    # `.float()` IS VERY IMPORTANT !!!
    # https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
    num_corrects = int(pred_flat.eq(target_flat.squeeze(1)).masked_select(mask_flat).float().data.sum())
    num_words = length.data.sum()

    return loss, pred_seqs, num_corrects, num_words.item()


def write_to_tensorboard(writer, global_step, total_loss, total_corrects, total_words, total_accuracy,
                         encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm,
                         encoder, decoder, gpu_memory_usage=None):
    # scalars
    if gpu_memory_usage is not None:
        writer.add_scalar('curr_gpu_memory_usage', gpu_memory_usage['curr'], global_step)
        writer.add_scalar('diff_gpu_memory_usage', gpu_memory_usage['diff'], global_step)

    writer.add_scalar('total_loss', total_loss, global_step)
    writer.add_scalar('total_accuracy', total_accuracy, global_step)
    writer.add_scalar('total_corrects', total_corrects, global_step)
    writer.add_scalar('total_words', total_words, global_step)
    writer.add_scalar('encoder_grad_norm', encoder_grad_norm, global_step)
    writer.add_scalar('decoder_grad_norm', decoder_grad_norm, global_step)
    writer.add_scalar('clipped_encoder_grad_norm', clipped_encoder_grad_norm, global_step)
    writer.add_scalar('clipped_decoder_grad_norm', clipped_decoder_grad_norm, global_step)

    # histogram
    for name, param in encoder.named_parameters():
        name = name.replace('.', '/')
        writer.add_histogram('encoder/{}'.format(name), variable2numpy(param), global_step, bins='doane')
        if param.grad is not None:
            writer.add_histogram('encoder/{}/grad'.format(name), variable2numpy(param.grad), global_step, bins='doane')

    for name, param in decoder.named_parameters():
        name = name.replace('.', '/')
        writer.add_histogram('decoder/{}'.format(name), variable2numpy(param), global_step, bins='doane')
        if param.grad is not None:
            writer.add_histogram('decoder/{}/grad'.format(name), variable2numpy(param.grad), global_step, bins='doane')


def detach_hidden(hidden):
    """ Wraps hidden states in new Variables, to detach them from their history. Prevent OOM.
        After detach, the hidden's requires_grad=Fasle and grad_fn=None.
    Issues:
    - Memory leak problem in LSTM and RNN: https://github.com/pytorch/pytorch/issues/2198
    - https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    - https://discuss.pytorch.org/t/help-clarifying-repackage-hidden-in-word-language-model/226
    - https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
    - 
    """
    if type(hidden) == Variable:
        hidden.detach()  # same as creating a new variable.
    else:
        for h in hidden: h.detach()


def train_batch(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens,
                encoder, decoder, encoder_optim, decoder_optim, opts):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    assert src_seqs.size(1) == tgt_seqs.size(1)
    batch_size = src_seqs.size(1)

    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = src_seqs.to(device)
    tgt_seqs = tgt_seqs.to(device)
    src_lens = torch.LongTensor(src_lens).to(device)
    tgt_lens = torch.LongTensor(tgt_lens).to(device)

    # Decoder's input
    input_seq = torch.LongTensor([BOS] * batch_size).to(device)

    # Decoder's output sequence length = max target sequence length of current batch.
    max_tgt_len = tgt_lens.data.max()

    # Store all decoder's outputs.
    # **CRUTIAL** 
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM, 
    # so we intialize tensor with fixed size instead:
    # `opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    decoder_outputs = torch.zeros(opts.max_seq_len, batch_size, decoder.vocab_size).to(device)

    # -------------------------------------
    # Training mode (enable dropout)
    # -------------------------------------
    encoder.train()
    decoder.train()

    # -------------------------------------
    # Zero gradients, since optimizers will accumulate gradients for every backward.
    # -------------------------------------
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    encoder_outputs, encoder_hidden = encoder(src_seqs, src_lens)

    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_hidden

    # Run through decoder one time step at a time.
    for t in range(max_tgt_len):
        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden,
                                                                    encoder_outputs, src_lens)

        # Store decoder outputs.
        decoder_outputs[t] = decoder_output

        # Next input is current target
        input_seq = tgt_seqs[t]

        # Detach hidden state:
        detach_hidden(decoder_hidden)

    # -------------------------------------
    # Compute loss
    # -------------------------------------
    loss, pred_seqs, num_corrects, num_words = masked_cross_entropy(
        decoder_outputs[:max_tgt_len].transpose(0, 1).contiguous(),
        tgt_seqs.transpose(0, 1).contiguous(),
        tgt_lens
    )

    pred_seqs = pred_seqs[:max_tgt_len]

    # -------------------------------------
    # Backward and optimize
    # -------------------------------------
    # Backward to get gradients w.r.t parameters in model.
    loss.backward()

    # Clip gradients
    encoder_grad_norm = nn.utils.clip_grad_norm(encoder.parameters(), opts.max_grad_norm)
    decoder_grad_norm = nn.utils.clip_grad_norm(decoder.parameters(), opts.max_grad_norm)
    clipped_encoder_grad_norm = compute_grad_norm(encoder.parameters())
    clipped_decoder_grad_norm = compute_grad_norm(decoder.parameters())

    # Update parameters with optimizers
    encoder_optim.step()
    decoder_optim.step()

    return loss.item(), pred_seqs, attention_weights, num_corrects, num_words, encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm


# You can download the small grammatical error correction dataset from [here](https://github.com/keisks/jfleg).
train_dataset = NMTDataset(src_path='../jfleg/dev/dev.src',
                           tgt_path='../jfleg/dev/dev.ref1')
# train_dataset = NMTDataset(src_path='../dataset/efcamdat/efcamdat2.changed.src.txt',
#                           tgt_path='../dataset/efcamdat/efcamdat2.changed.tgt.txt')
valid_dataset = NMTDataset(src_path='../jfleg/dev/dev.src',
                           tgt_path='../jfleg/dev/dev.ref0')

batch_size = 48
train_iter = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=collate_fn)
valid_iter = DataLoader(dataset=valid_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        collate_fn=collate_fn)

# Hyperparameters
# If enabled, load checkpoint.
LOAD_CHECKPOINT = False

opts = get_opts(LOAD_CHECKPOINT)

print('=' * 100)
print('Options log:')
print('- Load from checkpoint: {}'.format(LOAD_CHECKPOINT))
# todo
# if LOAD_CHECKPOINT:
#     print('- Global step: {}'.format(checkpoint['global_step']))
for k, v in opts.items():
    print('- {}: {}'.format(k, v))
print('=' * 100 + '\n')

# ### Initialize embeddings and models
# Initialize vocabulary size.
src_vocab_size = src_vocab_helper.vocab_size
tgt_vocab_size = tgt_vocab_helper.vocab_size

# Initialize embeddings.
# We can actually put all modules in one module like `NMTModel`)
# See: https://github.com/spro/practical-pytorch/issues/34
word_vec_size = opts.word_vec_size if not opts.pretrained_embeddings else nlp.vocab.vectors_length
src_embedding = nn.Embedding(src_vocab_size, word_vec_size, padding_idx=PAD)
tgt_embedding = nn.Embedding(tgt_vocab_size, word_vec_size, padding_idx=PAD)

if opts.share_embeddings:
    assert (src_vocab_size == tgt_vocab_size)
    tgt_embedding.weight = src_embedding.weight

# Initialize models.
encoder = EncoderRNN(embedding=src_embedding,
                     rnn_type=opts.rnn_type,
                     hidden_size=opts.hidden_size,
                     num_layers=opts.num_layers,
                     dropout=opts.dropout,
                     bidirectional=opts.bidirectional)
encoder = encoder.to(device)

decoder = LuongAttnDecoderRNN(encoder, embedding=tgt_embedding,
                              attention=opts.attention,
                              tie_embeddings=opts.tie_embeddings,
                              dropout=opts.dropout)
decoder = decoder.to(device)

if opts.pretrained_embeddings:
    glove_embeddings = load_spacy_glove_embedding(nlp, VocabHelper("./dataset/src_vocab.json"))
    encoder.embedding.weight.data.copy_(glove_embeddings)
    decoder.embedding.weight.data.copy_(glove_embeddings)
    if opts.fixed_embeddings:
        encoder.embedding.weight.requires_grad = False
        decoder.embedding.weight.requires_grad = False

# if LOAD_CHECKPOINT:
#     encoder.load_state_dict(checkpoint['encoder_state_dict'])
#     decoder.load_state_dict(checkpoint['decoder_state_dict'])

# ### Fine-tuning embeddings
# Recommend to use fine-tune after training for a while until the training loss don't decrease.
# 
# TODO: Should be controlled in training loop.

FINE_TUNE = True
if FINE_TUNE:
    encoder.embedding.weight.requires_grad = True

print('=' * 100)
print('Model log:\n')
print(encoder)
print(decoder)
print('- Encoder input embedding requires_grad={}'.format(encoder.embedding.weight.requires_grad))
print('- Decoder input embedding requires_grad={}'.format(decoder.embedding.weight.requires_grad))
print('- Decoder output embedding requires_grad={}'.format(decoder.W_s.weight.requires_grad))
print('=' * 100 + '\n')

# ### Initialize optimizers
# TODO: Different learning rate for fine tuning embeddings: https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/7
# Initialize optimizers (we can experiment different learning rates)
encoder_optim = optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=opts.learning_rate,
                           weight_decay=opts.weight_decay)
decoder_optim = optim.Adam([p for p in decoder.parameters() if p.requires_grad], lr=opts.learning_rate,
                           weight_decay=opts.weight_decay)

""" Open port 6006 and see tensorboard.
    Ref:  https://medium.com/@dexterhuang/%E7%B5%A6-pytorch-%E7%94%A8%E7%9A%84-tensorboard-bb341ce3f837
"""

# --------------------------
# Configure tensorboard
# --------------------------
model_name = 'seq2seq'
datetime = ('%s' % datetime.now()).split('.')[0]
experiment_name = '{}_{}'.format(model_name, datetime)
tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
writer = SummaryWriter(tensorboard_log_dir)


def train():
    # --------------------------
    # Configure training
    # --------------------------
    num_epochs = opts.num_epochs
    print_every_step = opts.print_every_step
    save_every_step = opts.save_every_step
    # For saving checkpoint and tensorboard
    # fixme:
    # global_step = 0 if not LOAD_CHECKPOINT else checkpoint['global_step']
    global_step = 0
    # --------------------------
    # Start training
    # --------------------------
    total_loss = 0
    total_corrects = 0
    total_words = 0
    prev_gpu_memory_usage = 0
    for epoch in range(num_epochs):
        for batch_id, batch_data in tqdm(enumerate(train_iter)):
            # Unpack batch data
            src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch_data

            # Ignore batch if there is a long sequence.
            max_seq_len = max(src_lens + tgt_lens)
            if max_seq_len > opts.max_seq_len:
                print('[!] Ignore batch: sequence length={} > max sequence length={}'.format(max_seq_len,
                                                                                             opts.max_seq_len))
                continue

            # Train.
            loss, pred_seqs, attention_weights, num_corrects, num_words, encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm = \
                train_batch(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, encoder, decoder,
                            encoder_optim,
                            decoder_optim, opts)

            # Statistics.
            global_step += 1
            total_loss += loss
            total_corrects += num_corrects
            total_words += num_words
            total_accuracy = 100 * (total_corrects / total_words)

            # Save checkpoint.
            if global_step % save_every_step == 0:
                checkpoint_dict = {
                    'opts': opts,
                    'global_step': global_step,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder_optim_state_dict': encoder_optim.state_dict(),
                    'decoder_optim_state_dict': decoder_optim.state_dict()
                }
                checkpoint_path = 'checkpoints/%s_acc_%.2f_loss_%.2f_step_%d.pt' % (
                    experiment_name, total_accuracy, total_loss, global_step)
                save_checkpoint(checkpoint_dict, checkpoint_path)

                print('=' * 100)
                print('Save checkpoint to "{}".'.format(checkpoint_path))
                print('=' * 100 + '\n')

            # Print statistics and write to Tensorboard.
            if global_step % print_every_step == 0:
                curr_gpu_memory_usage = get_gpu_memory_usage(device_id=torch.cuda.current_device())
                diff_gpu_memory_usage = curr_gpu_memory_usage - prev_gpu_memory_usage
                prev_gpu_memory_usage = curr_gpu_memory_usage

                print('=' * 100)
                print('Training log:')
                print('- Epoch: {}/{}'.format(epoch, num_epochs))
                print('- Global step: {}'.format(global_step))
                print('- Total loss: {}'.format(total_loss))
                print('- Total corrects: {}'.format(total_corrects))
                print('- Total words: {}'.format(total_words))
                print('- Total accuracy: {}'.format(total_accuracy))
                print('- Current GPU memory usage: {}'.format(curr_gpu_memory_usage))
                print('- Diff GPU memory usage: {}'.format(diff_gpu_memory_usage))
                print('=' * 100 + '\n')

                write_to_tensorboard(writer, global_step, total_loss, total_corrects, total_words, total_accuracy,
                                     encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm,
                                     clipped_decoder_grad_norm,
                                     encoder, decoder,
                                     gpu_memory_usage={'curr': curr_gpu_memory_usage, 'diff': diff_gpu_memory_usage})

                total_loss = 0
                total_corrects = 0
                total_words = 0

            # Free memory
            del src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, loss, pred_seqs, attention_weights, num_corrects, num_words, encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm

    checkpoint_dict = {
        'opts': opts,
        'global_step': global_step,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optim_state_dict': encoder_optim.state_dict(),
        'decoder_optim_state_dict': decoder_optim.state_dict()
    }
    checkpoint_path = 'checkpoints/%s_acc_%.2f_loss_%.2f_step_%d.pt' % (
        experiment_name, total_accuracy, total_loss, global_step)
    save_checkpoint(checkpoint_dict, checkpoint_path)
    print('=' * 100)
    print('Save checkpoint to "{}".'.format(checkpoint_path))
    print('=' * 100 + '\n')


# ## Evaluation
def evaluate_batch(src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, encoder, decoder):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)
    assert (batch_size == tgt_seqs.size(1))

    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = src_seqs.to(device)
    tgt_seqs = tgt_seqs.to(device)
    src_lens = torch.LongTensor(src_lens).to(device)
    tgt_lens = torch.LongTensor(tgt_lens).to(device)

    # Decoder's input
    input_seq = torch.LongTensor([BOS] * batch_size).to(device)

    # Decoder's output sequence length = max target sequence length of current batch.
    max_tgt_len = tgt_lens.data.max()

    # Store all decoder's outputs.
    # **CRUTIAL** 
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM, 
    # so we intialize tensor with fixed size instead:
    # `opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    decoder_outputs = torch.zeros(opts.max_seq_len, batch_size, decoder.vocab_size).to(device)

    # -------------------------------------
    # Evaluation mode (disable dropout)
    # -------------------------------------
    encoder.eval()
    decoder.eval()

    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    encoder_outputs, encoder_hidden = encoder(src_seqs, src_lens.data.tolist())

    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_hidden

    # Run through decoder one time step at a time.
    for t in range(max_tgt_len):
        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden,
                                                                    encoder_outputs, src_lens)
        # Store decoder outputs.
        decoder_outputs[t] = decoder_output
        # Next input is current target
        input_seq = tgt_seqs[t]
        # Detach hidden state (may not need this, since no BPTT)
        detach_hidden(decoder_hidden)

    # -------------------------------------
    # Compute loss
    # -------------------------------------
    loss, pred_seqs, num_corrects, num_words = masked_cross_entropy(
        decoder_outputs[:max_tgt_len].transpose(0, 1).contiguous(),
        tgt_seqs.transpose(0, 1).contiguous(),
        tgt_lens
    )

    pred_seqs = pred_seqs[:max_tgt_len]

    return loss.data[0], pred_seqs, attention_weights, num_corrects, num_words


def evaluate():
    total_loss = 0
    total_corrects = 0
    total_words = 0
    for batch_id, batch_data in tqdm(enumerate(valid_iter)):
        src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens = batch_data
        loss, pred_seqs, attention_weights, num_corrects, num_words = evaluate_batch(src_sents, tgt_sents, src_seqs,
                                                                                     tgt_seqs,
                                                                                     src_lens, tgt_lens, encoder,
                                                                                     decoder)

        total_loss += loss
        total_corrects += num_corrects
        total_words += num_words
        total_accuracy = 100 * (total_corrects / total_words)

    print('=' * 100)
    print('Validation log:')
    print('- Total loss: {}'.format(total_loss))
    print('- Total corrects: {}'.format(total_corrects))
    print('- Total words: {}'.format(total_words))
    print('- Total accuracy: {}'.format(total_accuracy))
    print('=' * 100 + '\n')


# ## Translate (Inference)
def translate_batch(src_text, train_dataset, encoder, decoder, max_seq_len, replace_unk=True):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Like dataset's `__getitem__()` and dataloader's `collate_fn()`.
    src_sent = src_text.split()
    src_seqs = torch.LongTensor([train_dataset.tokens2ids(tokens=src_text.split(),
                                                          token2id=train_dataset.src_vocab.token2id,
                                                          append_BOS=False, append_EOS=True)]).transpose(0, 1)
    src_lens = [len(src_seqs)]

    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)

    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = src_seqs.to(device)
    src_lens = torch.LongTensor(src_lens).to(device)

    # Decoder's input
    input_seq = torch.LongTensor([BOS] * batch_size).to(device)
    # Store output words and attention states
    out_sent = []
    all_attention_weights = torch.zeros(max_seq_len, len(src_seqs))

    # -------------------------------------
    # Evaluation mode (disable dropout)
    # -------------------------------------
    encoder.eval()
    decoder.eval()

    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    encoder_outputs, encoder_hidden = encoder(src_seqs, src_lens.data.tolist())

    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_hidden

    # Run through decoder one time step at a time.
    for t in range(max_seq_len):

        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_weights = decoder(input_seq, decoder_hidden,
                                                                    encoder_outputs, src_lens)

        # Store attention weights.
        # .squeeze(0): remove `batch_size` dimension since batch_size=1
        all_attention_weights[t] = attention_weights.squeeze(0).cpu().data

        # Choose top word from decoder's output
        prob, token_id = decoder_output.data.topk(1)
        token_id = token_id[0][0]  # get value
        if token_id == EOS:
            break
        else:
            if token_id == UNK and replace_unk:
                # Replace unk by selecting the source token with the highest attention score.
                score, idx = all_attention_weights[t].max(0)
                token = src_sent[idx[0]]
            else:
                # <UNK>
                token = train_dataset.tgt_vocab.id2token[token_id]

            out_sent.append(token)

        # Next input is chosen word
        input_seq = Variable(torch.LongTensor([token_id]), volatile=True).to(device)

        # Repackage hidden state (may not need this, since no BPTT)
        detach_hidden(decoder_hidden)

    src_text = ' '.join([train_dataset.src_vocab.id2token[token_id] for token_id in src_seqs.data.squeeze(1).tolist()])
    out_text = ' '.join(out_sent)

    # all_attention_weights: (out_len, src_len)
    return src_text, out_text, all_attention_weights[:len(out_sent)]


def translate_demo():
    # ### Small test for translation
    src_text, out_text, all_attention_weights = translate_batch('He have a car', train_dataset, encoder, decoder,
                                                                max_seq_len=opts.max_seq_len)
    src_text, out_text, all_attention_weights

    # check attention weight sum == 1
    [all_attention_weights[t].sum() for t in range(all_attention_weights.size(0))]


def translate_file(in_file='../jfleg/test/test.src'):
    # ### Translate a given text file
    test_src_texts = []
    with codecs.open(in_file, 'r', 'utf-8') as f:
        test_src_texts = f.readlines()

    test_src_texts[:5]

    out_texts = []
    for src_text in test_src_texts:
        _, out_text, _ = translate_batch(src_text.strip(), train_dataset, encoder, decoder,
                                         max_seq_len=opts.max_seq_len)
        out_texts.append(out_text)

    out_texts[:5]

    # ### Save the predictions to text file
    with codecs.open('./pred.txt', 'w', 'utf-8') as f:
        for text in out_texts:
            f.write(text + '\n')


def main():
    train()
    evaluate()
    translate_demo()
    translate_file()


if __name__ == '__main__':
    main()

# ### Evaluate with GLEU metric
# If you're playing with grammatical error correction (GEC) corpus (jfleg),
# it has an evaluation script specifically for GEC task:
# 
# Run:
# ```
# python jfleg/eval/gleu.py \
# -s jfleg/test/test.src \
# -r jfleg/test/test.ref[0-3] \
# --hyp ./pred.txt
# ```
# 
# Output (GLEU score, std, confidence interval):
# Note: The OpenNMT-py can further achieves ~0.49 GLEU score with the same model settings.
# TODO: Try to optimize the code.
# ```
# Running GLEU...
# ./pred.txt
# [['0.451747', '0.007620', '(0.437,0.467)']]
# ```

# ### Notes:
# - Set `MAX_LENGTH` to training sequence is important to prevent OOM.
#     - Will effect：`decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))`
# - Do not `next(iter(data_loader))` in training for-loop，could be very slow.
# - When computing `num_corrects`, need to cast `ByteTensor` using `.float()` in order to do `.sum()`, otherwise the result will overflow. Ref: https://discuss.pytorch.org/t/batch-size-and-validation-accuracy/4066/3
# - Very crutial to GPU memory usage: Don'T set `MAX_LENGTH` to `max(tgt_lens)`. Varying tensor size could cause GPU allocate a new memory, so we fixed tensor size instead: `decoder_outputs = Variable(torch.zeros(**MAX_LENGTH**, batch_size, decoder.vocab_size))`
# - Be careful if you only want to get `Variable`'s data and do some operations, for example, `sum()`, you should use `Variable(...).data.sum()` instead of `Variable(...).sum().data[0]`. This will create a new computational graph and if you do this in for-loop, it might increase memory.
# - Be careful to misuse `Variable`.
# - Do `detach` for RNN's hidden states, or it might increase memory when doing backprop.
# - If restart but GPU memory is not returned, kill all python processes: `>> ps x |grep python|awk '{print $1}'|xargs kill`
# - Forward decoder is time-consuming (for-loop).
# - Calling `backward()` free memory: https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735
# 
# ### Try to:
# - Implement schedule sampling for training.
# - Implement beam search for evaluation and translation.
# - Understand and interpret param visualization on tensorboard.
# - Implement more RNN optimizing and regularization tricks:
#     - Set `max_seq_len` for preventing RNN OOM 
#     - Xavier initializer
#     - Weight normalization and layer normalization: https://github.com/pytorch/pytorch/issues/1601
#     - Embedding dropout
#     - Weight dropping
#     - Variational dropout: [part1](https://becominghuman.ai/learning-note-dropout-in-recurrent-networks-part-1-57a9c19a2307), [part2](https://towardsdatascience.com/learning-note-dropout-in-recurrent-networks-part-2-f209222481f8), [part3](https://towardsdatascience.com/learning-note-dropout-in-recurrent-networks-part-3-1b161d030cd4)
#     - Zoneout
#     - Fraternal dropout
#     - Activation regularization (AR), and temporal activation regularization (TAR)
#     - Read more: [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf)
