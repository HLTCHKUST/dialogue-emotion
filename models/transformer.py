### TAKEN FROM https://github.com/kolloldas/torchnlp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
from models.common_layer import EncoderLayer ,DecoderLayer ,MultiHeadAttention ,Conv ,PositionwiseFeedForward ,LayerNorm ,_gen_bias_mask ,_gen_timing_signal
from utils import constant
from utils.features import share_embedding

class UTransformer(nn.Module):
    """
    A Transformer Module For BabI data. 
    Inputs should be in the shape story: [batch_size, memory_size, story_len ]
                                  query: [batch_size, 1, story_len]
    Outputs will have the shape [batch_size, ]
    """
    def __init__(self, vocab, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=700, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, act=False ):
        super(UTransformer, self).__init__()
        self.embedding_dim = embedding_size
        self.vocab = vocab
        self.emb = share_embedding(self.vocab,constant.pretrained,fix_pretrain=constant.fix_pretrain)
        self.transformer_enc = Encoder(embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                                filter_size, max_length=max_length, input_dropout=input_dropout, layer_dropout=layer_dropout, 
                                attention_dropout=attention_dropout, relu_dropout=relu_dropout, use_mask=False, act=act)

        self.W = nn.Linear(self.embedding_dim,4) ## 4 emotion
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, X_lengths=None):
        ## QUERY ENCODER + MUlt Mask
        X = self.emb(X)

        ## APPLY TRANSFORMER
        logit, act = self.transformer_enc(X)
        
        a_hat = self.W(torch.sum(logit,dim=1)/logit.size(1)) ## reduce mean

        return a_hat, self.softmax(a_hat), act

class HUTransformer(nn.Module):
    """
    A Transformer Module For BabI data. 
    Inputs should be in the shape story: [batch_size, memory_size, story_len ]
                                  query: [batch_size, 1, story_len]
    Outputs will have the shape [batch_size, ]
    """
    def __init__(self, vocab, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=700, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, act=False ):
        super(HUTransformer, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_size
        self.emb = share_embedding(self.vocab,constant.pretrained,fix_pretrain=constant.fix_pretrain)
        self.transformer_enc = Encoder(embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                                filter_size, max_length=max_length, input_dropout=input_dropout, layer_dropout=layer_dropout, 
                                attention_dropout=attention_dropout, relu_dropout=relu_dropout, use_mask=False, act=act)
        self.transformer_hier = Encoder(embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                                filter_size, max_length=3, input_dropout=input_dropout, layer_dropout=layer_dropout, 
                                attention_dropout=attention_dropout, relu_dropout=relu_dropout, use_mask=False, act=act) 
        self.W = nn.Linear(self.embedding_dim,4) ## 4 emotion
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X_1,X_2,X_3, x1_len,x2_len,x3_len, extract_feature=False):
        ## QUERY ENCODER + MUlt Mask
        X_1_emb = self.emb(X_1)
        X_2_emb = self.emb(X_2)
        X_3_emb = self.emb(X_3)

        ## APPLY TRANSFORMER
        logit_1 = torch.sum(self.transformer_enc(X_1_emb, X_1 if constant.mask else None )[0],dim=1)
        logit_2 = torch.sum(self.transformer_enc(X_2_emb, X_2 if constant.mask else None )[0],dim=1)
        logit_3 = torch.sum(self.transformer_enc(X_3_emb, X_3 if constant.mask else None )[0],dim=1)

        logit = self.transformer_hier(torch.stack([logit_1,logit_2,logit_3],dim=1))[0]
        
        a_hat = self.W(torch.sum(logit,dim=1)/logit.size(1)) ## reduce mean


        # a_hat = self.W(torch.concat(logit,dim=1)) ## reduce mean
        if extract_feature:
            return logit[:,2,:]
        a_hat = self.W(logit[:,2,:]) ## reduce mean



        return a_hat, self.softmax(a_hat), None


# def share_embedding(vocab, pretrain=True):
#     embedding = nn.Embedding(vocab.n_words, constant.emb_dim, padding_idx=constant.PAD_idx)
#     if(pretrain):
#         pre_embedding = gen_embeddings(vocab,emb_file = constant.emb_file)
#         embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
#         embedding.weight.data.requires_grad = False
#     return embedding

class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, act=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        # device = torch.device('cuda:{}'.format(constant.device))
        # model  
        self.timing_signal = _gen_timing_signal(max_length, hidden_size).to(constant.device)
        ## for t
        self.position_signal = _gen_timing_signal(num_layers, hidden_size).to(constant.device)

        self.num_layers = num_layers
        self.act = act
        
        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)

        self.proj_flag = False
        if not (embedding_size == hidden_size):
            self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
            self.proj_flag = True

        if(constant.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.Sequential(*[EncoderLayer(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        if(self.act):
            self.act_fn = ACT_basic(hidden_size)

    def forward(self, inputs, lengths=None):
        if(lengths is None):
            mask = None
        else:
            w_batch, w_len = lengths.size()
            mask = lengths.data.eq(constant.PAD_idx).unsqueeze(1).expand(w_batch, w_len, w_len)

        #Add input dropout
        x = self.input_dropout(inputs)
        if(self.proj_flag):
            # Project to hidden size
            x = self.embedding_proj(x)

        if(constant.universal):
            if(self.act):
                print("here")
                x, (remainders,n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                return x, (remainders,n_updates)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x = self.enc(x, src_mask=mask)
                # x = self.layer_norm(x)
                return x, None
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            y = self.enc(x)
        
            y = self.layer_norm(y)
            return y, None
        

# def get_attn_key_pad_mask(seq_k, seq_q):
#     ''' For masking out the padding part of key sequence. '''
#     # Expand to fit the shape of key query attention matrix.
#     len_q = seq_q.size(1)
#     PAD = 0
#     padding_mask = seq_k.eq(PAD)
#     padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

#     return padding_mask



### CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
class ACT_basic(nn.Module):
    def __init__(self,hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)  
        self.p.bias.data.fill_(1) 
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S
        remainders = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()
        step = 0
        # for l in range(self.num_layers):
        while( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if(encoder_output):
                state, _ = fn((state,encoder_output))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        return previous_state, (remainders,n_updates)
