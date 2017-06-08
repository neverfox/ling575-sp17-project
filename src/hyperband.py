# Copyright (c) 2017, Zygmunt Zając
# All rights reserved.

# This software is licensed as follows:

# 1. All Russian, Chinese and United States government agencies are strictly
# prohibited from using this software.

# 2. For other entities the license is BSD-2-clause, as follows.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from math import ceil, log
from pprint import pprint
from random import random
from time import ctime, time

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable, backward
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm

import utils
from conll_u import CoNLLU, HeadField
from model import Parser
from torchtext import data

try:
    from hyperopt import hp
    from hyperopt.pyll.stochastic import sample
except ImportError:
    print(
        "In order to achieve operational capability, this programme requires hyperopt to be installed (pip install hyperopt), unless you make get_params() use something else."
    )


class Hyperband:
    def __init__(self, get_params_function, try_params_function):
        self.get_params = get_params_function
        self.try_params = try_params_function

        self.max_iter = 27  # maximum iterations per configuration
        self.eta = 3  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta**s))

            # initial number of iterations per config
            r = self.max_iter * self.eta**(-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta**(-i)
                n_iterations = r * self.eta**(i)

                print("\n*** {} configurations x {:.1f} iterations each".
                      format(n_configs, n_iterations))

                val_losses = []
                early_stops = []

                for t in T:

                    self.counter += 1
                    print("\n{} | {} | best so far: {:.4f} (run {})\n".format(
                        self.counter,
                        ctime(), self.best_loss, self.best_counter))

                    start_time = time()

                    if dry_run:
                        result = {
                            'loss': random(),
                            'log_loss': random(),
                            'auc': random()
                        }
                    else:
                        result = self.try_params(n_iterations, t)  # <---

                    assert (type(result) == dict)
                    assert ('loss' in result)

                    seconds = int(round(time() - start_time))
                    print("\n{} seconds.".format(seconds))

                    loss = result['loss']
                    val_losses.append(loss)

                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append(result)

                    torch.save(self.results,
                               os.path.join("../output",
                                            config.lang + ".params.pt"))

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[0:int(n_configs / self.eta)]

        return self.results


space = {
    'batch_size': hp.quniform('bs', 1, 10, 1),
    'd_hidden': hp.qloguniform('h', log(128), log(768), 128),
    'd_mlp': hp.qloguniform('m', log(128), log(512), 128),
    'n_layers': hp.quniform('l', 1, 3, 1),
    'dp_ratio': hp.quniform('dp', 0, 0.5, 0.05),
    'clip': hp.qlognormal('c', log(5), log(10), 1),
    'lr': hp.quniform('lr', 0.001, 0.002, 0.0001),
    'lr_decay': hp.quniform('lrd', 0.25, 1, 0.05),
    'beta_2': hp.qloguniform('b2', log(0.9), log(0.99), 0.01),
    'shared_mlp': hp.choice('sm', [True, False]),
    'nonlinear_attn': hp.choice('nl', [True, False])
}

# Parse arguments
args = utils.get_args(sys.argv[1:])
config = args
config.device = -1

# Set the random seed manually for reproducibility
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    if not config.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )
    else:
        print("Congratulations! You have a CUDA device and it will be used!")
        config.device = 0
        torch.cuda.manual_seed(config.seed)
else:
    print("Unfortunately, you don't have a CUDA device. Brew a pot of coffee.")

###############################################################################
# Load data
###############################################################################

FORM = data.Field(init_token='<root>', lower=config.lower)
LEMMA = data.Field(init_token='<root>')
POS = data.Field(init_token='<root>')
HEAD = HeadField()
DEPREL = data.Field()

train, val, test = CoNLLU.splits(
    config.lang,
    FORM,
    LEMMA,
    POS,
    HEAD,
    DEPREL,
    dropbox_root='/LING 575 Project Data')

# Build vocabularies
FORM.build_vocab(
    train,
    val,
    test,
    wv_dir='../input',
    wv_type=args.lang,
    wv_dim=args.d_embed)
LEMMA.build_vocab(train, val, test)
POS.build_vocab(train, val, test)
DEPREL.build_vocab(train)

config.n_embed_form = len(FORM.vocab)
config.n_embed_lemma = len(LEMMA.vocab)
config.n_embed_pos = len(POS.vocab)
config.n_deprel = len(DEPREL.vocab)
config.padding_idx = FORM.vocab.stoi['<pad>']


def handle_integers(params):
    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v

    return new_params


def get_params():
    params = sample(space)
    return handle_integers(params)


def try_params(n_iterations, t):
    config.batch_size = t['batch_size']
    config.d_hidden = t['d_hidden']
    config.d_mlp = t['d_mlp']
    config.n_layers = t['n_layers']
    config.dp_ratio = t['dp_ratio']
    config.clip = t['clip']
    config.lr = t['lr']
    config.lr_decay = t['lr_decay']
    config.beta_2 = t['beta_2']
    config.shared_mlp = t['shared_mlp']
    config.nonlinear_attn = t['nonlinear_attn']

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test),
        batch_sizes=(config.batch_size, 1, 1),
        device=args.device,
        repeat=False)

    model = Parser(config)
    if config.use_pre:
        model.embed_form.weight.data.copy_(FORM.vocab.vectors)
        if config.fix_embed:
            model.embed.weight.requires_grad = False
    if config.cuda:
        model.cuda()

    print('Model architecture:')
    print('-' * 89)
    print(model)
    print('-' * 89)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta_1, config.beta_2),
        weight_decay=config.weight_decay)

    best_loss = np.inf
    for epoch in np.arange(1, int(round(n_iterations)) + 1):
        try:
            train_epoch(model, optimizer, train_iter, epoch, config)
            loss = val_epoch(model, val_iter, epoch, config)
        except:
            loss = np.inf
        if loss == float("-inf"):
            loss == np.inf
        best_loss = min(loss, best_loss)

    return {'loss': best_loss}


def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 1"""
    lr = config.lr * pow(config.lr_decay, epoch / config.epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def prep_batch(batch, config):
    heads, lengths = batch.head
    batch_size = heads.size()[0]
    seq_len = heads.size()[1]
    heads = heads.unsqueeze(3).expand(batch_size, seq_len, seq_len,
                                      config.n_deprel)
    labels = batch.deprel.t().clone().cpu().data.numpy()
    labels = (
        np.arange(config.n_deprel) == labels[:, :, None]).astype(np.uint8)
    labels = np.expand_dims(labels, 1)
    labels = np.repeat(labels, seq_len, 1)
    labels = np.insert(labels, 0, np.zeros(config.n_deprel), 2)
    labels = torch.from_numpy(labels)
    if config.device != -1:
        labels = labels.cuda()
    target = torch.mul(heads.data, labels)
    batch.target = Variable(target)
    return batch


def train_epoch(model, optimizer, batch_iter, epoch, config):
    # Turn on training mode to enable dropout
    model.train()

    batch_iter.init_epoch()

    desc = 'Training epoch {}'.format(epoch)

    total_loss = 0
    for batch in tqdm(batch_iter, desc=desc, leave=False):
        batch = prep_batch(batch, config)

        # Clear accumulated gradients
        optimizer.zero_grad()

        # Get losses
        energies = model.energies(batch)
        losses = model.loss(batch, energies)

        # Backpropaganda
        backward([loss for loss in losses])
        clip_grad_norm(model.parameters(), config.clip)
        optimizer.step()

        total_loss += torch.mean(losses).data[0]

    avg_loss = total_loss / len(batch_iter)

    lr = adjust_learning_rate(optimizer, epoch, config)

    print('Trained epoch {} - Loss: {:.4f}, Learning Rate: {:.6f}'.format(
        epoch, avg_loss, lr))


def eval_batches(model, batch_iter, config):
    # Turn on eval mode to disable dropout
    model.eval()

    total_loss = 0
    for batch in batch_iter:
        batch = prep_batch(batch, config)
        energies = model(batch)
        losses = model.loss(batch, energies).data
        total_loss += torch.sum(losses)

    return total_loss / len(batch_iter)


def val_epoch(model, batch_iter, epoch, config):
    desc = 'Validating epoch {}'.format(epoch)
    avg_loss = eval_batches(
        model, tqdm(
            batch_iter, desc=desc, leave=False), config)
    print('Validated epoch {} - Loss: {:.4f}'.format(epoch, avg_loss))
    return avg_loss


def main():
    output_file = os.path.join("../output", config.lang + ".params.pt")

    hb = Hyperband(get_params, try_params)
    results = hb.run(skip_last=1)

    print("{} total, best:\n".format(len(results)))

    for r in sorted(results, key=lambda x: x['loss'])[:5]:
        print("loss: {:.4f} | {} seconds | {:.1f} iterations | run {} ".format(
            r['loss'], r['seconds'], r['iterations'], r['counter']))
        pprint(r['params'])
        print()

    print("saving...")

    torch.save(results, output_file)


if __name__ == '__main__':
    main()
