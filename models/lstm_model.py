import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

import numpy as np
import math

from models.common_layer import Attention
from utils import constant
from models.common_layer import MultiHeadAttention, BertPooler
from utils.features import share_embedding

class LstmModel(nn.Module):
    """
    An LSTM model. 
    Inputs: 
        X: (batch_size, seq_len)
        X_lengths: (batch_size)
    Outputs: (batch_size, labels)
    """
    def __init__(self, vocab, embedding_size, hidden_size, num_layers, max_length=700, input_dropout=0.0, layer_dropout=0.0, is_bidirectional=False, attentive=False):
        super(LstmModel, self).__init__()
        self.embedding_dim = embedding_size
        self.hidden_size = hidden_size

        self.input_dropout = nn.Dropout(input_dropout)
        self.layer_dropout = nn.Dropout(layer_dropout)
        self.vocab = vocab
        # self.emb = nn.Embedding(num_vocab, embedding_size, padding_idx=0)
        self.emb = share_embedding(self.vocab,constant.pretrained, fix_pretrain=constant.fix_pretrain)
        self.lstm = nn.LSTM(embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional)
        self.W = nn.Linear(hidden_size*2 if is_bidirectional else hidden_size,4) ## 4 emotion
        self.softmax = nn.Softmax(dim=1)

        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

    def forward(self, X, X_lengths, extract_feature=False):
        """
        Forward algorithm
        if extract_feature is True: returns output of LSTM before output layer
        else: returns the logits and softmax of logits
        """
        X = self.emb(X)
        X = self.input_dropout(X)

        X = X.transpose(0, 1) # (len, batch_size, dim)

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=False)
        _, hidden = self.lstm(packed_input) 
        # returns hidden state of all timesteps as well as hidden state at last timestep
        # should take last non zero hidden state, not last timestamp (may have zeros), don't take output
        # output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)

        last_hidden = hidden[-1] # (num_direction, batch_size, dim)
        if len(last_hidden.size()) == 3:
            last_hidden_1 = last_hidden[-1]

        if self.is_bidirectional:
            last_hidden = torch.cat((last_hidden[0].squeeze(0), last_hidden[1].squeeze(0)), dim=1)
        else:
            last_hidden = last_hidden.squeeze(0)

        if extract_feature:
            return last_hidden

        last_hidden = self.layer_dropout(last_hidden)

        a_hat = self.W(last_hidden) # (batch_size, 4)
        return a_hat, self.softmax(a_hat)

class HLstmModel(nn.Module):
    """
    A Hierarchical LSTM model with self-attention.
    Inputs: 
        X_1: (batch_size, seq_len),
        X_2: (batch_size, seq_len),
        X_3: (batch_size, seq_len),
        X_lengths: (batch_size)
    Outputs: (batch_size, labels)
    """
    def __init__(self, vocab, embedding_size, hidden_size, num_layers, max_length=700, input_dropout=0.0, layer_dropout=0.0, is_bidirectional=False,
     attentive=False, multiattentive=False, num_heads=5, total_key_depth=500, total_value_depth=1000, use_mask = True, sum_tensor=False, super_ratio=0.0, double_supervision=False,
     context=False,pool="",pool_stride=2, pool_kernel=3):
        super(HLstmModel, self).__init__()
        self.embedding_dim = embedding_size
        self.hidden_size = hidden_size
        self.sum_tensor = sum_tensor
        self.input_dropout = nn.Dropout(input_dropout)
        self.layer_dropout = nn.Dropout(layer_dropout)
        self.vocab = vocab

        self.pooler = BertPooler(hidden_size*2 if is_bidirectional else hidden_size)
        # self.emb = nn.Embedding(num_vocab, embedding_size, padding_idx=0)
        self.emb = share_embedding(self.vocab,constant.pretrained, constant.fix_pretrain)
        self.context = context
        if self.context:
            self.context_lstm = nn.LSTM(embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional)

        self.lstm = nn.LSTM(embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional)
        
        self.pool = pool
        if self.pool == "avg":
            kernel_size = pool_kernel
            stride = pool_stride
            padding = 0

            l_in = hidden_size
            if is_bidirectional:
                l_in *= 2

            self.pooling_layer = nn.AvgPool1d(kernel_size, stride=stride, padding=padding)
            l_out = int(((l_in + 2 * padding - kernel_size) / stride) + 1)

            self.lstm_layer = nn.LSTM(l_out, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional)
        elif self.pool == "max":
            kernel_size = pool_kernel
            stride = pool_stride
            padding = 0
            dilation = 1

            l_in = hidden_size
            if is_bidirectional:
                l_in *= 2

            self.pooling_layer = nn.MaxPool1d(kernel_size, stride=stride, padding=padding, dilation=dilation)
            l_out = int(((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

            self.lstm_layer = nn.LSTM(l_out, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional)
        elif self.pool == "globalmax" or self.pool == "globalavg":
            self.lstm_layer = nn.LSTM(hidden_size*2 if is_bidirectional else hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional)
        else:
            self.lstm_layer = nn.LSTM(hidden_size*2 if is_bidirectional else hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional)
        self.W = nn.Linear(hidden_size*2 if is_bidirectional else hidden_size,4) ## 4 emotion
        self.softmax = nn.Softmax(dim=1)

        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

        self.attentive =attentive
        self.multiattentive = multiattentive
        self.use_mask = use_mask
        if self.multiattentive:
            self.attentive = True
        if self.attentive:
            if self.multiattentive:
                self.word_attention = MultiHeadAttention(hidden_size*2 if is_bidirectional else hidden_size, total_key_depth, total_value_depth,
                hidden_size*2 if is_bidirectional else hidden_size, num_heads)
                self.sentences_attention = MultiHeadAttention(hidden_size*2 if is_bidirectional else hidden_size, total_key_depth, total_value_depth,
                hidden_size*2 if is_bidirectional else hidden_size, num_heads)
            else:
                self.word_attention = Attention(hidden_size*2 if is_bidirectional else hidden_size)
                self.sentences_attention = Attention(hidden_size*2 if is_bidirectional else hidden_size)

        self.double_supervision = double_supervision
        if double_supervision:
            self.output_super = nn.Linear(hidden_size*3, 4) # lower layer supervision

    def forward(self, X_1, X_2, X_3, X_1_lengths, X_2_lengths, X_3_lengths, extract_feature=False, cuda=True):
        """
        Forward algorithm
        if extract_feature is True: returns output of LSTM before output layer
        else: returns the logits and softmax of logits
        """
        if self.use_mask:
            mask1 = X_1.data.eq(constant.PAD_idx).unsqueeze(1)
            mask2 = X_2.data.eq(constant.PAD_idx).unsqueeze(1)
            mask3 = X_3.data.eq(constant.PAD_idx).unsqueeze(1)
        else:
            mask1=mask2=mask3=None
        X_1 = self.emb(X_1)
        X_2 = self.emb(X_2)
        X_3 = self.emb(X_3)

        # sort X_2 and X_3
        X_1_lengths = torch.LongTensor(X_1_lengths)
        sorted_X_2_lengths, perm_index_x_2 = torch.sort(torch.LongTensor(X_2_lengths), descending=True)
        sorted_X_3_lengths, perm_index_x_3 = torch.sort(torch.LongTensor(X_3_lengths), descending=True)
        if cuda:
            X_1_lengths = X_1_lengths.cuda()
            sorted_X_2_lengths = sorted_X_2_lengths.cuda()
            sorted_X_3_lengths = sorted_X_3_lengths.cuda()
            perm_index_x_2 = perm_index_x_2.cuda()
            perm_index_x_3 = perm_index_x_3.cuda()

        sorted_X_2 = X_2[perm_index_x_2]
        sorted_X_3 = X_3[perm_index_x_3]
        
        X_1 = self.input_dropout(X_1).transpose(0, 1) # (len, batch_size, dim)
        sorted_X_2 = self.input_dropout(sorted_X_2).transpose(0, 1) # (len, batch_size, dim)
        sorted_X_3 = self.input_dropout(sorted_X_3).transpose(0, 1) # (len, batch_size, dim)

        # returns hidden state of all timesteps as well as hidden state at last timestep
        # should take last non zero hidden state, not last timestamp (may have zeros), don't take output
        packed_input_1 = torch.nn.utils.rnn.pack_padded_sequence(X_1, X_1_lengths, batch_first=False)
        packed_input_2 = torch.nn.utils.rnn.pack_padded_sequence(sorted_X_2, sorted_X_2_lengths, batch_first=False)
        packed_input_3 = torch.nn.utils.rnn.pack_padded_sequence(sorted_X_3, sorted_X_3_lengths, batch_first=False)
        
        if self.context:
            lstm_out1, hidden_1 = self.context_lstm(packed_input_1) # hidden: (len, batch_size, dim)
            lstm_out2, hidden_2 = self.context_lstm(packed_input_2) # hidden: (len, batch_size, dim)
        else:
            lstm_out1, hidden_1 = self.lstm(packed_input_1) # hidden: (len, batch_size, dim)
            lstm_out2, hidden_2 = self.lstm(packed_input_2) # hidden: (len, batch_size, dim)
        lstm_out3, hidden_3 = self.lstm(packed_input_3) # hidden: (len, batch_size, dim)

        if self.attentive:
            padded_lstm_out1, _ = nn.utils.rnn.pad_packed_sequence(lstm_out1)
            padded_lstm_out2, _ = nn.utils.rnn.pad_packed_sequence(lstm_out2)
            padded_lstm_out3, _ = nn.utils.rnn.pad_packed_sequence(lstm_out3)
            padded_lstm_out1 = padded_lstm_out1.permute(1,0,2) #(batch, seq_len, num_directions * hidden_size)
            padded_lstm_out2 = padded_lstm_out2.permute(1,0,2)
            padded_lstm_out3 = padded_lstm_out3.permute(1,0,2)
            if self.multiattentive:
                
                last_hidden_1 = self.word_attention(padded_lstm_out1,padded_lstm_out1,padded_lstm_out1,src_mask=mask1)
                last_hidden_2 = self.word_attention(padded_lstm_out2,padded_lstm_out2,padded_lstm_out2,src_mask=mask2)
                last_hidden_3 = self.word_attention(padded_lstm_out3,padded_lstm_out3,padded_lstm_out3,src_mask=mask3)
                if self.sum_tensor:
                    last_hidden_1 = torch.sum(last_hidden_1, dim=1)
                    last_hidden_2 = torch.sum(last_hidden_2, dim=1)
                    last_hidden_3 = torch.sum(last_hidden_3, dim=1)
                else:
                    last_hidden_1 = self.pooler(last_hidden_1)
                    last_hidden_2 = self.pooler(last_hidden_2)
                    last_hidden_3 = self.pooler(last_hidden_3)
            else:
                last_hidden_1 = self.word_attention(padded_lstm_out1, X_1_lengths) #(batch, num_directions * hidden_size)
                last_hidden_2 = self.word_attention(padded_lstm_out2, sorted_X_2_lengths)
                last_hidden_3 = self.word_attention(padded_lstm_out3, sorted_X_2_lengths)
        else:
            last_hidden_1 = hidden_1[-1] # (num_direction * num_layer, batch_size, dim)
            last_hidden_2 = hidden_2[-1] # (num_direction * num_layer, batch_size, dim)
            last_hidden_3 = hidden_3[-1] # (num_direction * num_layer, batch_size, dim)

            batch_size = last_hidden_1.size(1)
            dim = last_hidden_1.size(2)
            last_hidden_1 = last_hidden_1.view(self.num_layers, 2 if self.is_bidirectional else 1, batch_size, dim)[-1]
            last_hidden_2 = last_hidden_2.view(self.num_layers, 2 if self.is_bidirectional else 1, batch_size, dim)[-1]
            last_hidden_3 = last_hidden_3.view(self.num_layers, 2 if self.is_bidirectional else 1, batch_size, dim)[-1]

            if self.is_bidirectional:
                last_hidden_1 = torch.cat((last_hidden_1[0].squeeze(0), last_hidden_1[1].squeeze(0)), dim=1)
                last_hidden_2 = torch.cat((last_hidden_2[0].squeeze(0), last_hidden_2[1].squeeze(0)), dim=1)
                last_hidden_3 = torch.cat((last_hidden_3[0].squeeze(0), last_hidden_3[1].squeeze(0)), dim=1)
            else:
                last_hidden_1 = last_hidden_1.squeeze(0) # (batch_size, dim)
                last_hidden_2 = last_hidden_2.squeeze(0) # (batch_size, dim)
                last_hidden_3 = last_hidden_3.squeeze(0) # (batch_size, dim)

        # restore the order
        unsorted_last_hidden_2 = last_hidden_2.new(*last_hidden_2.size())
        unsorted_last_hidden_2.scatter_(0, perm_index_x_2.unsqueeze(1).expand(last_hidden_2.size(0), last_hidden_2.size(1)), last_hidden_2)
        unsorted_last_hidden_3 = last_hidden_3.new(*last_hidden_3.size())
        unsorted_last_hidden_3.scatter_(0, perm_index_x_3.unsqueeze(1).expand(last_hidden_3.size(0), last_hidden_3.size(1)), last_hidden_3)

        last_hidden = torch.cat((last_hidden_1.unsqueeze(0), unsorted_last_hidden_2.unsqueeze(0), unsorted_last_hidden_3.unsqueeze(0)), dim=0) # (3, batch_size, dim)
        concatenated_hidden = last_hidden.transpose(0, 1) # (batch_size, 3, dim)
        if self.pool == "avg" or self.pool == "max":
            last_hidden = self.pooling_layer(last_hidden)
        elif self.pool == "globalavg" or self.pool == "globalmax":
            context_hidden = last_hidden[:2] # (2, batch_size, dim)
            if self.pool == "globalavg":
                context_hidden = context_hidden.mean(dim=0).unsqueeze(0) # (1, batch_size, dim)
            else:
                context_hidden = context_hidden.max(dim=0)[0].unsqueeze(0) # (1, batch_size, dim)
            turn3_hidden = last_hidden[2].unsqueeze(0) # (1, batch_size, dim)
            last_hidden = torch.cat((context_hidden, turn3_hidden), dim=0)

        concatenated_hidden = concatenated_hidden.contiguous().view(concatenated_hidden.size(0), -1) # (batch_size, 3 * dim)

        lstm_layer_out, hidden = self.lstm_layer(last_hidden)
        if self.attentive:
            lstm_layer_out = lstm_layer_out.permute(1,0,2) #(batch, seq_len, num_directions * hidden_size)
            if self.multiattentive:
                last_hidden = torch.sum(self.sentences_attention(lstm_layer_out, lstm_layer_out, lstm_layer_out),dim=1)
            else:
                last_hidden = self.sentences_attention(lstm_layer_out) #(batch, num_directions * hidden_size)
        else:
            last_hidden = hidden[-1] # (num_direction * num_layers, batch_size, dim)
            batch_size = last_hidden.size(1)
            dim = last_hidden.size(2)
            last_hidden = last_hidden.view(self.num_layers, 2 if self.is_bidirectional else 1, batch_size, dim)[-1]

            if self.is_bidirectional:
                last_hidden = torch.cat((last_hidden[0].squeeze(0), last_hidden[1].squeeze(0)), dim=1)
            else:
                last_hidden = last_hidden.squeeze(0)

        if extract_feature: # for SVM
            if self.double_supervision:
                return torch.cat((concatenated_hidden, last_hidden), dim=1)
            else:
                return last_hidden

        last_hidden = self.layer_dropout(last_hidden)

        a_hat = self.W(last_hidden) # (batch_size, 4) 
        if self.double_supervision:
            additional_logits = self.output_super(concatenated_hidden) # (batch_size, 4)
            return a_hat, self.softmax(a_hat), additional_logits, self.softmax(additional_logits)
        return a_hat, self.softmax(a_hat)
