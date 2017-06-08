import os
import re
import sys
from argparse import ArgumentParser
from collections import Counter


class ConllEntry:
    def __init__(self,
                 id,
                 form,
                 lemma,
                 pos,
                 cpos,
                 feats=None,
                 parent_id=None,
                 relation=None,
                 deps=None,
                 misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [
            str(self.id), self.form, self.lemma, self.cpos, self.pos,
            self.feats, str(self.pred_parent_id)
            if self.pred_parent_id is not None else None, self.pred_relation,
            self.deps, self.misc
        ]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordsCount.update([
                node.norm for node in sentence if isinstance(node, ConllEntry)
            ])
            posCount.update([
                node.pos for node in sentence if isinstance(node, ConllEntry)
            ])
            relCount.update([
                node.relation for node in sentence
                if isinstance(node, ConllEntry)
            ])

    return (wordsCount, {w: i
                         for i, w in enumerate(wordsCount.keys())},
            posCount.keys(), relCount.keys())


def read_conll(fh):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1,
                      'rroot', '_', '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1:
                yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tokens.append(
                    ConllEntry(
                        int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5],
                        int(tok[6])
                        if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def get_args(args=sys.argv[1:]):
    parser = ArgumentParser(description='Dependency Parser')
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--d_hidden', type=int, default=200)
    parser.add_argument('--d_mlp', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--dp_ratio', type=int, default=0.33)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay', type=float, default=.75)
    parser.add_argument('--beta_1', type=float, default=.9)
    parser.add_argument('--beta_2', type=float, default=.9)
    parser.add_argument('--weight_decay', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--preserve_case', action='store_false', dest='lower')
    parser.add_argument('--fix_embed', action='store_true', dest='fix_embed')
    parser.add_argument(
        '--concat_embed', action='store_true', dest='concat_embed')
    parser.add_argument(
        '--nonlinear_attn', action='store_true', dest='nonlinear_attn')
    parser.add_argument('--no_birnn', action='store_false', dest='birnn')
    parser.add_argument('--no_pre', action='store_false', dest='use_pre')
    parser.add_argument('--no_pos', action='store_false', dest='use_pos')
    parser.add_argument('--no_lemma', action='store_false', dest='use_lemma')
    parser.add_argument('--shared_mlp', action='store_true', dest='shared_mlp')
    parser.add_argument('--cuda', action='store_true', dest='cuda')
    parser.add_argument('--save_path', type=str, default='../output')
    parser.add_argument('--resume', action='store_true', dest='resume')
    parser.add_argument('--start_epoch', type=int, default=1)
    args = parser.parse_args(args)
    return args
