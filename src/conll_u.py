import os
import sys
from collections import defaultdict

import numpy as np
import torch
from conllu.parser import parse
from scipy import sparse
from torch.autograd import Variable

from torchtext import data

lang_dict = {'en': 'English'}


def fromCoNLLU(sentence, fields):
    nd = defaultdict(list)
    for d in sentence:
        for key, val in d.items():
            nd[key].append(val)
    ex = data.Example.fromdict(dict(nd.items()), fields)
    return ex


class HeadField(data.Field):
    def __init__(self):
        super(HeadField, self).__init__(
            use_vocab=False,
            tensor_type=torch.ByteTensor,
            include_lengths=True)

    def pad(self, minibatch):
        minibatch = list(minibatch)
        max_len = max(len(x) for x in minibatch)
        padded, lengths = [], []
        for x in minibatch:
            padded.append(([] if self.init_token is None else
                           [self.init_token]) + list(x[:max_len]) +
                          ([] if self.eos_token is None else [self.eos_token]
                           ) + [None] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        return (padded, lengths)

    def numericalize(self, arr, device=None, train=True):
        arr, lengths = arr
        masks = []
        for x, n in list(zip(arr, lengths)):
            shape = (len(x) + 1, ) * 2
            xys = [(h, m) for h, m in zip(x, np.arange(1, n + 1))
                   if None.__ne__(h)]
            coords = tuple(zip(*xys))
            mask = sparse.coo_matrix(
                (np.ones(len(coords[0])), coords), shape=shape,
                dtype=np.uint8).toarray()
            masks.append(mask.tolist())
        arr = self.tensor_type(masks)
        lengths = torch.LongTensor(lengths)
        if device != -1:
            arr = arr.cuda(device)
            lengths = lengths.cuda(device)
        return Variable(arr, volatile=not train), lengths


class CoNLLU(data.Dataset):
    def __init__(self, examples, fields, **kwargs):
        return super(CoNLLU, self).__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.form)

    @classmethod
    def splits(cls, lang, form_field, lemma_field, pos_field, head_field,
               deprel_field, **kwargs):
        fname_pt = os.path.join('../input', lang + '_splits.pt')
        if os.path.isfile(fname_pt):
            print('Loading splits from', fname_pt)
            train, validation, test = torch.load(fname_pt)
        else:
            train_fname = os.path.join('ud-treebanks-conll2017',
                                       'UD_' + lang_dict[lang],
                                       lang + '-ud-train.conllu')
            val_fname = os.path.join('ud-treebanks-conll2017',
                                     'UD_' + lang_dict[lang],
                                     lang + '-ud-dev.conllu')
            test_fname = os.path.join('ud-test-v2.0-conll2017', 'gold',
                                      'conll17-ud-test-2017-05-09',
                                      lang + '.conllu')

            if os.path.isfile(train_fname):
                print('Loading training data from', train_fname)
                train = parse(open(train_fname, 'rb'))
            else:
                print('Cannot find any training data')
                sys.exit()

            if os.path.isfile(val_fname):
                print('Loading validation data from', val_fname)
                validation = parse(open(val_fname, 'rb'))
            else:
                print('Cannot find any validation data')
                sys.exit()

            if os.path.isfile(test_fname):
                print('Loading test data from', test_fname)
                test = parse(open(test_fname, 'rb'))
            else:
                print('Cannot find any test data')
                sys.exit()

            torch.save((train, validation, test), fname_pt)

        field_dict = {
            'form': ('form', form_field),
            'lemma': ('lemma', lemma_field),
            'upostag': ('pos', pos_field),
            'head': ('head', head_field),
            'deprel': ('deprel', deprel_field)
        }

        fields = [field for field in field_dict.values()]

        train = cls([fromCoNLLU(s, field_dict) for s in train], fields,
                    **kwargs)
        validation = cls([fromCoNLLU(s, field_dict) for s in validation],
                         fields, **kwargs)
        test = cls([fromCoNLLU(s, field_dict) for s in test], fields, **kwargs)

        return train, validation, test

    @classmethod
    def iters(cls,
              lang,
              batch_size=32,
              device=-1,
              wv_dir='../input',
              wv_type='en',
              wv_dim=100,
              **kwargs):
        FORM = data.Field(init_token='<root>')
        LEMMA = data.Field(init_token='<root>')
        POS = data.Field(init_token='<root>')
        HEAD = HeadField()
        DEPREL = data.Field()

        train, val, test = cls.splits(lang, FORM, LEMMA, POS, HEAD, DEPREL,
                                      **kwargs)

        FORM.build_vocab(train, wv_dir=wv_dir, wv_type=wv_type, wv_dim=wv_dim)
        LEMMA.build_vocab(train)
        POS.build_vocab(train)
        HEAD.build_vocab(train)
        DEPREL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, val, test), batch_size=batch_size, device=device)
