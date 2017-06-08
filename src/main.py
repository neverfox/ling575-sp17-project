import os
import shutil
import sys

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

print('Neural Dependency Parser')

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

train, val, test = CoNLLU.splits(config.lang, FORM, LEMMA, POS, HEAD, DEPREL)

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

train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train, val, test),
    batch_sizes=(config.batch_size, 1, 1),
    device=args.device,
    repeat=False)

config.n_embed_form = len(FORM.vocab)
config.n_embed_lemma = len(LEMMA.vocab)
config.n_embed_pos = len(POS.vocab)
config.n_deprel = len(DEPREL.vocab)
config.padding_idx = FORM.vocab.stoi['<pad>']

###############################################################################
# Build the model
###############################################################################

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


def save_checkpoint(state,
                    is_best,
                    filename=os.path.join(config.save_path,
                                          config.lang + '.checkpoint.pt')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(
            filename,
            os.path.join(config.save_path, config.lang + '.model.pt'))


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
    total_uas = 0
    total_las = 0
    for batch in batch_iter:
        batch = prep_batch(batch, config)
        energies = model(batch)
        losses = model.loss(batch, energies).data
        (pred, _, _) = model.predict(batch, energies)
        target = batch.target.data
        lengths = batch.head[1]

        uas_pred = pred.sum(3, keepdim=False)
        uas_target = target.sum(3, keepdim=False)
        uas_corr = (uas_pred * uas_target).sum(1, keepdim=False).sum(
            1, keepdim=False)

        las_corr = (pred * target).sum(1, keepdim=False).sum(
            1, keepdim=False).sum(1, keepdim=False)
        uas = 100. * uas_corr.float() / lengths.float()
        las = 100. * las_corr.float() / lengths.float()

        total_loss += torch.sum(losses)
        total_uas += torch.sum(uas)
        total_las += torch.sum(las)

    return total_loss / len(batch_iter), total_uas / len(
        batch_iter), total_las / len(batch_iter)


def val_epoch(model, batch_iter, epoch, config):
    desc = 'Validating epoch {}'.format(epoch)
    avg_loss, avg_uas, avg_las = eval_batches(
        model, tqdm(
            batch_iter, desc=desc, leave=False), config)
    print('Validated epoch {} - Loss: {:.4f}, UAS: {:.3f}, LAS: {:.3f}'.format(
        epoch, avg_loss, avg_uas, avg_las))
    return avg_loss, avg_las


def test_model(model, batch_iter, config):
    desc = 'Testing model'
    avg_loss, avg_uas, avg_las = eval_batches(
        model, tqdm(
            batch_iter, desc=desc, leave=False), config)
    print('Tested model - Loss: {:.4f}, UAS: {:.3f}, LAS: {:.3f}'.format(
        avg_loss, avg_uas, avg_las))


def train():
    print('Training for {} epochs with batch size {}'.format(
        config.epochs, config.batch_size))

    best_las = 0

    if config.resume:
        fname = os.path.join(config.save_path, config.lang + '.checkpoint.pt')
        if os.path.isfile(fname):
            print("=> loading checkpoint {}".format(fname))
            checkpoint = torch.load(fname)
            config.start_epoch = checkpoint['epoch']
            best_las = checkpoint['best_las']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(fname, checkpoint['epoch']))
        else:
            print("No checkpoint found. Starting from the top.")

    try:
        for epoch in np.arange(config.start_epoch, config.epochs + 1):
            train_epoch(model, optimizer, train_iter, epoch, config)
            loss, las = val_epoch(model, val_iter, epoch, config)
            is_best = not best_las or las > best_las
            best_las = max(las, best_las)
            save_checkpoint({
                'epoch': epoch + 1,
                'best_las': best_las,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def test():
    try:
        test_model(model, test_iter, config)
    except KeyboardInterrupt:
        sys.exit()


def main():
    train()

    best_fname = os.path.join(config.save_path, config.lang + '.model.pt')
    if os.path.isfile(best_fname):
        checkpoint = torch.load(best_fname)
        best_epoch = checkpoint['epoch'] - 1
        best_las = checkpoint['best_las']
        model.load_state_dict(checkpoint['state_dict'])
        print("Loading best model from epoch {} with LAS {:.3f}".format(
            best_epoch, best_las))

    test()


if __name__ == '__main__':
    main()
