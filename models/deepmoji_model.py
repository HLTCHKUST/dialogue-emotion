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
from utils.features import share_embedding

from utils.emo_features import MojiModel, EmoFeatures

from torchmoji.model_def import torchmoji_feature_encoding
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

class HDeepMoji(nn.Module):
    """
    A stack of DeepMoji-LSTM model with self-attention.
    Inputs: 
        X_1: (batch_size, seq_len),
        X_2: (batch_size, seq_len),
        X_3: (batch_size, seq_len),
        X_lengths: (batch_size)
    Outputs: (batch_size, labels)
    """
    def __init__(self, vocab, hidden_size, num_layers, max_length=700, input_dropout=0.0, layer_dropout=0.0, is_bidirectional=False, attentive=False, multiattentive=True, num_heads=5, total_key_depth=500, total_value_depth=1000, use_mask = True):
        super(HDeepMoji, self).__init__()

        self.input_dropout = nn.Dropout(input_dropout)
        self.layer_dropout = nn.Dropout(layer_dropout)
        self.vocab = vocab

        self.torchmoji = torchmoji_feature_encoding(PRETRAINED_PATH)
        embedding_size = 2304
        # self.lstm = nn.LSTM(embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional, batch_first=False)
        self.W = nn.Linear(hidden_size*2 if is_bidirectional else hidden_size,4) ## 4 emotion
        self.softmax = nn.Softmax(dim=1)

        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

        self.use_mask = use_mask
        self.attentive = attentive
        if attentive:
            # self.word_attention = Attention(hidden_size*2 if is_bidirectional else hidden_size)
            self.sentences_attention = Attention(hidden_size*2 if is_bidirectional else hidden_size)

    def forward(self, X_1, X_2, X_3, X_1_lengths, X_2_lengths, X_3_lengths, cuda=True):
        # print("X_1", X_1)
        if self.use_mask: # mask for attention
            mask1 = X_1.data.eq(constant.PAD_idx).unsqueeze(1)
            mask2 = X_2.data.eq(constant.PAD_idx).unsqueeze(1)
            mask3 = X_3.data.eq(constant.PAD_idx).unsqueeze(1)
        else:
            mask1=mask2=mask3=None

        X_1 = self.torchmoji(X_1) # return vector
        X_2 = self.torchmoji(X_2)
        X_3 = self.torchmoji(X_3)

        X_1 = self.input_dropout(X_1)
        X_2 = self.input_dropout(X_2)
        X_3 = self.input_dropout(X_3)

        last_hidden_1 = X_1
        unsorted_last_hidden_2 = X_2
        unsorted_last_hidden_3 = X_3

        last_hidden = torch.cat((last_hidden_1.unsqueeze(0), unsorted_last_hidden_2.unsqueeze(0), unsorted_last_hidden_3.unsqueeze(0)), dim=0) # (3, batch_size, dim)
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

        last_hidden = self.layer_dropout(last_hidden)

        a_hat = self.W(last_hidden) # (batch_size, 4)
        return a_hat, self.softmax(a_hat)