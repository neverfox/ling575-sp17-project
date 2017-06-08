import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.init as init

from eval import decode_MST
from linalg import logdet


class Bottle(nn.Module):
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BiBottle(nn.Module):
    def forward(self, input1, input2):
        assert len(input1.size()) == len(input2.size(
        )), "Inputs have different number of dimensions"
        assert input1.size()[0] == input2.size()[
            0], "Inputs have different first dimension size"

        if len(input1.size()) <= 2:
            return super(BiBottle, self).forward(input1, input2)

        size1 = input1.size()[:2]
        size2 = input2.size()[:2]

        assert size1[1] == size2[
            1], "Inputs have different second dimension size"

        out = super(BiBottle, self).forward(
            input1.view(size1[0] * size1[1], -1),
            input2.view(size2[0] * size2[1], -1))
        return out.view(size1[0], size1[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Bilinear(BiBottle, nn.Bilinear):
    pass


class Encoder(nn.Module):
    """
    Shape:
        - Input: `(batch, seq_len, input_size)`
        - Output: `(batch, seq_len, out_size)`
    """

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config

        if self.config.concat_embed:
            input_size = self.config.d_embed * 3
        else:
            input_size = self.config.d_embed

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=self.config.d_hidden,
            num_layers=self.config.n_layers,
            dropout=self.config.dp_ratio,
            bidirectional=self.config.birnn)

        self.n_dir = 2 if self.config.birnn else 1
        self.d_rnn_out = self.config.d_hidden * self.n_dir
        self.n_cells = self.config.n_layers * self.n_dir

        if config.shared_mlp:
            self.mlp = nn.Sequential(
                Linear(self.d_rnn_out, self.config.d_mlp),
                nn.ReLU(),
                nn.Dropout(p=self.config.dp_ratio))
        else:
            self.mlp_h = nn.Sequential(
                Linear(self.d_rnn_out, self.config.d_mlp),
                nn.ReLU(),
                nn.Dropout(p=self.config.dp_ratio))
            self.mlp_m = nn.Sequential(
                Linear(self.d_rnn_out, self.config.d_mlp),
                nn.ReLU(),
                nn.Dropout(p=self.config.dp_ratio))

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        output, (h_t, c_t) = self.rnn(inputs, self.init_hidden(batch_size))
        if self.config.shared_mlp:
            feats = self.mlp(output.permute(1, 0, 2).contiguous())
            return feats
        else:
            feats_h = self.mlp_h(output.permute(1, 0, 2).contiguous())
            feats_m = self.mlp_m(output.permute(1, 0, 2).contiguous())
            return feats_h, feats_m

    def init_hidden(self, batch_size):
        state_shape = self.n_cells, batch_size, self.config.d_hidden
        h_0 = autograd.Variable(torch.zeros(*state_shape))
        c_0 = autograd.Variable(torch.zeros(*state_shape))
        if self.config.device != -1:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        return h_0, c_0


class Attention(nn.Module):
    """
    Shape:
        - Input: `(batch, seq_len, d_mlp)`
        - Output: `(batch, seq_len, seq_len, n_deprel)`
    """

    def __init__(self, config):
        super(Attention, self).__init__()

        self.config = config

        self.U = torch.Tensor(config.d_mlp, config.d_mlp, config.n_deprel)
        init.xavier_uniform(self.U, gain=init.calculate_gain('linear'))

        self.W_h = torch.Tensor(config.d_mlp, config.n_deprel)
        init.xavier_uniform(self.W_h, gain=init.calculate_gain('linear'))

        self.W_m = torch.Tensor(config.d_mlp, config.n_deprel)
        init.xavier_uniform(self.W_m, gain=init.calculate_gain('linear'))

        self.b = torch.zeros(config.n_deprel)

        if config.device != -1:
            self.U = self.U.cuda()
            self.W_h = self.W_h.cuda()
            self.W_m = self.W_m.cuda()
            self.b = self.b.cuda()

        self.U = nn.Parameter(self.U)
        self.W_h = nn.Parameter(self.W_h)
        self.W_m = nn.Parameter(self.W_m)
        self.b = nn.Parameter(self.b)

    def forward(self, inputs):
        if self.config.shared_mlp:
            batch_size = inputs.size()[0]
            inputs_m = inputs
            inputs_h_2d = inputs_m_2d = inputs.view(-1, self.config.d_mlp)
        else:
            inputs_h, inputs_m = inputs
            batch_size = inputs_h.size()[0]
            inputs_h_2d = inputs_h.view(-1, self.config.d_mlp)
            inputs_m_2d = inputs_m.view(-1, self.config.d_mlp)

        out = torch.mm(inputs_h_2d, self.U.view(self.config.d_mlp, -1)).view(
            batch_size, -1, self.config.d_mlp, self.config.n_deprel)

        seq_len = out.size()[1]

        out = torch.bmm(
            out.permute(0, 1, 3, 2).contiguous().view(batch_size, -1,
                                                      self.config.d_mlp),
            inputs_m.permute(0, 2, 1)).view(batch_size, seq_len,
                                            self.config.n_deprel,
                                            seq_len).permute(0, 1, 3, 2)

        s_h = torch.mm(inputs_h_2d, self.W_h).view(batch_size, seq_len, -1)

        out += s_h.unsqueeze(2).expand_as(out)

        s_m = torch.mm(inputs_m_2d, self.W_m).view(batch_size, seq_len, -1)

        out += s_m.unsqueeze(2).expand_as(out)

        out += self.b.expand_as(out)

        if self.config.nonlinear_attn:
            out = nn.ReLU()(out)

        return out


class Parser(nn.Module):
    def __init__(self, config):
        super(Parser, self).__init__()

        self.config = config

        # Embeddings
        self.embed_form = nn.Embedding(
            config.n_embed_form,
            config.d_embed,
            padding_idx=config.padding_idx)
        self.embed_lemma = nn.Embedding(
            config.n_embed_lemma,
            config.d_embed,
            padding_idx=config.padding_idx)
        self.embed_pos = nn.Embedding(
            config.n_embed_pos, config.d_embed, padding_idx=config.padding_idx)

        self.embed_dropout = nn.Dropout(p=config.dp_ratio)

        self.encoder = Encoder(config)
        self.attention = Attention(config)

    def energies(self, sentences):
        embeddings_form = self.embed_form(sentences.form)
        embeddings_lemma = self.embed_lemma(sentences.lemma)
        embeddings_pos = self.embed_pos(sentences.pos)
        if self.config.concat_embed:
            embeddings = torch.cat(
                [embeddings_form, embeddings_lemma, embeddings_pos], 2)
        else:
            embeddings = embeddings_form + embeddings_lemma + embeddings_pos

        embeddings = self.embed_dropout(embeddings)

        encodings = self.encoder(embeddings)
        energies = self.attention(encodings)

        return energies

    def loss(self, sentences, energies):
        """
        The forward pass for training, using CRF
        """
        input_shape = energies.size()
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        (_, lengths) = sentences.head
        target = sentences.target

        targets = autograd.Variable(torch.zeros(batch_size))
        if self.config.device != -1:
            targets = targets.cuda()

        for i in np.arange(batch_size):
            targets[i] = energies[i].masked_select(target[i]).sum()

        I1 = 1 - torch.eye(seq_len, seq_len)
        I1 = autograd.Variable(
            I1.unsqueeze(0).unsqueeze(3).expand_as(energies))
        if self.config.device != -1:
            I1 = I1.cuda()

        potentials = torch.mul(torch.exp(energies), I1)
        # sum over labels
        potentials = potentials.sum(3, keepdim=False)

        partitions = autograd.Variable(torch.zeros(batch_size))
        if self.config.device != -1:
            partitions = partitions.cuda()

        for b in np.arange(batch_size):
            # true length + root
            n = lengths[b] + 1
            # remove padding
            A = potentials[b, :n, :n]
            D = torch.sum(A, 0, keepdim=False)
            D = D.unsqueeze(1).expand_as(A)
            I2 = torch.eye(n, n)
            if self.config.device != -1:
                I2 = I2.cuda()
            I2 = autograd.Variable(I2)
            D = torch.mul(D, I2)
            D += D * 1e-6 + 1e-8
            L = D - A
            # Must move to CPU for custom function
            L_hat = L[1:, 1:].cpu()
            partition = logdet(L_hat)
            # Back to GPU
            if self.config.device != -1:
                partition = partition.cuda()
            partitions[b] = partition

        neg_log_likelihood = partitions - targets

        return neg_log_likelihood

    def predict(self, sentences, energies):
        _, lengths = sentences.head
        energies = energies.data.clone().cpu()
        lengths = lengths.clone().cpu()
        pred, heads, deprels = decode_MST(energies, lengths)
        pred = torch.from_numpy(pred)

        if self.config.cuda:
            pred = pred.cuda()

        return pred, heads, deprels

    def forward(self, sentences):
        """
        The forward pass for test instances, using MST decoder
        """
        energies = self.energies(sentences)

        return energies
