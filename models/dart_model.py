import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dart_genotypes import STEPS
from utils.dart_utils import mask2d
from utils.dart_utils import LockedDropout
from utils.dart_utils import embedded_dropout
from torch.autograd import Variable

from utils import dart_constant
from utils.features import gen_embeddings

constant = dart_constant

INITRANGE = 0.04


class DARTSCell(nn.Module):

    def __init__(self, embedding_size, hidden_size, dropouth, dropoutx, genotype=None):
        super(DARTSCell, self).__init__()
        self.hidden_size = hidden_size
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.genotype = genotype

        # genotype is None when doing arch search
        steps = len(
            self.genotype.recurrent) if self.genotype is not None else STEPS
        self._W0 = nn.Parameter(torch.FloatTensor(
            embedding_size+hidden_size, 2*hidden_size).uniform_(-INITRANGE, INITRANGE))
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(hidden_size, 2*hidden_size).uniform_(-INITRANGE, INITRANGE)) for i in range(steps)
        ])

    def forward(self, inputs, hidden=None):
        T, B = inputs.size(0), inputs.size(1)

        if hidden is None:
            hidden = inputs.new_zeros(*inputs.size(), requires_grad=False)

        if self.training:
            x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx)
            h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)
        else:
            x_mask = h_mask = None

        hidden = hidden[0]
        hiddens = []
        for t in range(T):
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens)
        return hiddens, hiddens[-1].unsqueeze(0)

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        # print(x.size(), x_mask.size(), h_prev.size(), h_mask.size())
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.hidden_size, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0-h_prev)
        return s0

    def _get_activation(self, name):
        if name == 'tanh':
            f = torch.tanh
        elif name == 'relu':
            f = torch.relu
        elif name == 'sigmoid':
            f = torch.sigmoid
        elif name == 'identity':
            def f(x): return x
        else:
            raise NotImplementedError
        return f

    def cell(self, x, h_prev, x_mask, h_mask):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

        states = [s0]
        for i, (name, pred) in enumerate(self.genotype.recurrent):
            s_prev = states[pred]
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[i])
            else:
                ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(name)
            h = fn(h)
            s = s_prev + c * (h-s_prev)
            states += [s]
        output = torch.mean(torch.stack(
            [states[i] for i in self.genotype.concat], -1), -1)
        return output


def share_embedding(vocab, pretrain=True):
    embedding = nn.Embedding(
        vocab.n_words, constant.emb_dim, padding_idx=constant.PAD_idx)
    if(pretrain):
        pre_embedding = gen_embeddings(vocab, emb_file=constant.emb_file)
        embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
        embedding.weight.data.requires_grad = False
    return embedding


class RNNModel(nn.Module):
    """
    An LSTM model. 
    Inputs: 
        X: (batch_size, seq_len)
        X_lengths: (batch_size)
    Outputs: (batch_size, labels)
    """

    def __init__(self, vocab, embedding_size, hidden_size, hidden_size_last, dropouth=0.5, dropoutx=0.5, num_layers=1, max_length=700, is_bidirectional=False, 
    cell_cls=DARTSCell, genotype=None):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()

        self.embedding_dim = embedding_size
        self.hidden_size = hidden_size

        self.vocab = vocab

        assert embedding_size == hidden_size == hidden_size_last
        if cell_cls == DARTSCell:
            assert genotype is not None
            self.rnns = [
                cell_cls(embedding_size, hidden_size, dropouth, dropoutx, genotype).cuda()]
        else:
            assert genotype is None
            # print(cell_cls)
            # print(">>>",embedding_size, hidden_size, dropouth, dropoutx)
            self.rnns = [
                cell_cls(embedding_size, hidden_size, dropouth, dropoutx).cuda()]

        # self.emb = nn.Embedding(num_vocab, embedding_size, padding_idx=0)
        self.emb = share_embedding(self.vocab, constant.preptrained)
        self.lstm = nn.LSTM(embedding_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=is_bidirectional)
        self.W = nn.Linear(
            hidden_size*2 if is_bidirectional else hidden_size, 4)  # 4 emotion
        self.softmax = nn.Softmax(dim=1)

        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

    def forward(self, X, hidden):
    # def forward(self, X, X_lengths):
        X = self.emb(X)
        # X = self.input_dropout(X)
        batch_size = X.size(0)

        # X = X.transpose(0, 1)  # (len, batch_size, dim)
        # if hidden.size(0) != X.size(0):
        #     hidden = hidden.transpose(0, 1)

        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(
        #     X, X_lengths, batch_first=False)

        # hidden = self.init_hidden(batch_size, self.hidden_size)
        # hidden = hidden[0]

        # if(constant.USE_CUDA):
        #     hidden = hidden.cuda()
        # print(hidden.size())
        raw_output, hidden = self.rnns[0](X, hidden)
        last_hidden = hidden[-1]
        # _, hidden = self.lstm(packed_input)

        # returns hidden state of all timesteps as well as hidden state at last timestep
        # should take last non zero hidden state, not last timestamp (may have zeros), don't take output
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)

        # print(raw_output)

        # last_hidden = hidden[-1]  # (num_direction, batch_size, dim)
        # if self.is_bidirectional:
        #     last_hidden = torch.cat(
        #         (last_hidden[0].squeeze(0), last_hidden[1].squeeze(0)), dim=1)
        # else:
        #     last_hidden = last_hidden.squeeze(0)

        # last_hidden = self.layer_dropout(last_hidden)

        a_hat = self.W(last_hidden)  # (batch_size, 4)
        return a_hat, self.softmax(a_hat)

# class RNNModel(nn.Module):
#     """Container module with an encoder, a recurrent module, and a decoder."""

#     def __init__(self, ntoken, nlabels, ninp, nhid, nhidlast,
#                  dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1, vocab=None,
#                  cell_cls=DARTSCell, genotype=None):
#         super(RNNModel, self).__init__()
#         # self.lockdrop = LockedDropout()
#         # self.encoder = nn.Embedding(ntoken, ninp)
#         self.encoder = share_embedding(vocab, constant.preptrained)
#         self.nlabels = nlabels

#         # assert ninp == nhid == nhidlast
#         # if cell_cls == DARTSCell:
#         #     assert genotype is not None
#         #     self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)]
#         # else:
#         #     assert genotype is None
#         #     self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx)]

#         # self.rnns = torch.nn.ModuleList(self.rnns)
#         self.decoder = nn.Linear(ninp, 4)
#         # self.decoder.weight = self.encoder.weight, size of them are different
#         # self.init_weights()

#         self.ninp = ninp
#         self.nhid = nhid
#         self.nhidlast = nhidlast
#         self.dropout = dropout
#         self.dropouti = dropouti
#         self.dropoute = dropoute
#         self.ntoken = ntoken
#         self.cell_cls = cell_cls

#         self.lstm = torch.nn.LSTM(ninp, nhid, batch_first=False)
#         self.softmax = nn.Softmax(dim=1)

#     # def init_weights(self):
#         # self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
#         # self.decoder.bias.data.fill_(0)
#         # self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

#     def forward(self, input, hidden, return_h=False):
#         input = input.transpose(0, 1)
#         # hidden = hidden.transpose(0, 1)
#         batch_size = input.size(1)

#         emb = self.encoder(input)

#         # emb = embedded_dropout(self.encoder, input,
#         #                        dropout=self.dropoute if self.training else 0)
#         # emb = self.lockdrop(emb, self.dropouti)

#         raw_output = emb
#         new_hidden = []
#         raw_outputs = []
#         outputs = []

#         _, hidden = self.lstm(emb)
#         hidden = hidden[-1] # (num_direction, batch_size, dim)
#         output = hidden

#         # for l, rnn in enumerate(self.rnns):
#         #     # print(">>hidden:", hidden[l])
#         #     current_input = raw_output
#         #     # raw_output, new_h = rnn(raw_output, hidden[l])
#         #     raw_output, new_h = self.lstm(raw_output, None)
#         #     new_h = new_h[-1]

#         #     new_hidden.append(new_h)
#         #     # print(new_h.size())
#         #     # print(">>>>>>>>>>>>>", raw_output[-1,:,:].size())
#         #     # raw_outputs.append(raw_output[-1,:,:])
#         #     raw_outputs.append(raw_output)
#         # hidden = new_hidden

#         # output = self.lockdrop(new_h, self.dropout)
#         # output = new_h
#         # output = self.lockdrop(raw_output, self.dropout)
#         # print("################ output:", output)

#         # logit = self.decoder(output)
#         logit = self.decoder(output.view(-1, self.ninp))
#         log_prob = self.softmax(logit)
#         model_output = log_prob
#         # print(">>>>>>>>>>>>>>>>>>>>>", model_output.size(), batch_size, self.nlabels)
#         model_output = model_output.view(-1, batch_size, self.nlabels)

#         if return_h:
#             return model_output, hidden, raw_outputs, outputs
#         return model_output, hidden

#     def init_hidden(self, bsz):
#         weight = next(self.parameters()).data
#         return [Variable(weight.new(1, bsz, self.nhid).zero_())]

    def init_hidden(self, batch_size, hidden_size):
        weight = next(self.parameters()).data
        return [weight.new_zeros(1, batch_size, hidden_size).zero_()]
