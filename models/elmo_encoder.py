import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.common_layer import Attention, MultiHeadAttention

from allennlp.modules.elmo import Elmo, batch_to_ids

from utils import constant

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class Attention(nn.Module):
    """
    Computes a weighted average of channels across timesteps (1 parameter pr. channel).
    aka. self-attention
    """

    def __init__(self, attention_size, return_attention=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
            return_attention: If true, output will include the weight for each input token
                              used for the prediction
        """
        super(Attention, self).__init__()
        self.return_attention = return_attention
        self.attention_size = attention_size
        self.attention_vector = nn.Parameter(torch.FloatTensor(attention_size))
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(attention_size,attention_size)

        nn.init.uniform_(self.attention_vector.data, -0.01, 0.01)

    def forward(self, inputs, input_lengths=None):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Tensor): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            Tuple with (representations and attentions if self.return_attention else None).
        """
        inputs = self.tanh(self.linear(inputs))
        logits = inputs.matmul(self.attention_vector)
        unnorm_ai = (logits - logits.max()).exp()

        # Compute a mask for the attention on the padded sequences
        # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        max_len = unnorm_ai.size(1)

        if input_lengths is not None:
            idxes = torch.arange(
                0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)

            if constant.USE_CUDA:
                idxes = idxes.cuda()

            mask = (idxes < input_lengths.unsqueeze(1)).float()
            
            # apply mask and renormalize attention scores (weights)
            masked_weights = unnorm_ai * mask
        else:
            masked_weights = unnorm_ai
        att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        attentions = masked_weights.div(att_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(dim=1)
        return representations

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, 
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from 
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)
                
            
        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5
        self.bias_mask = bias_mask
        
        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)
    
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)
        
    def forward(self, queries, keys, values, src_mask=None):
        
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        
        # Split into multiple heads
        queries = self._split_heads(queries) #[batch_size, num_heads, seq_length, depth/num_heads]
        keys = self._split_heads(keys) #[batch_size, num_heads, seq_length, depth/num_heads]
        values = self._split_heads(values) #[batch_size, num_heads, seq_length, depth/num_heads]
        
        # Scale queries
        queries *= self.query_scale
        
        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2)) # [batch_size, num_heads, seq_length, seq_length]

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).expand_as(logits)
            logits = logits.masked_fill(src_mask, -np.inf).type_as(logits.data)

        # Add bias to mask future values
        if self.bias_mask is not None:
            logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)
        
        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)
        
        # Dropout
        weights = self.dropout(weights)
        
        # Combine with values to get context
        contexts = torch.matmul(weights, values)
        
        # Merge heads
        contexts = self._merge_heads(contexts)
        #contexts = torch.tanh(contexts)
        
        # Linear to get output
        outputs = self.output_linear(contexts)
        
        return outputs


class ELMoEncoder(nn.Module):
    """
    ELMo + Linear
    Inputs: 
        x: (batch_size, seq_len, 50),
    Outputs: (batch_size, C)
    """
    def __init__(self, C, H, dropout=0.5):
        super(ELMoEncoder, self).__init__()
        self.elmo = Elmo(options_file, weight_file, 1, dropout=dropout)
        self.linear = nn.Linear(H, C)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x : flattened dialogs

        # run through ELMo ScalarMix
        h = self.elmo(x)['elmo_representations'][0]
        # print(h.shape)

        # sum dim=1
        h = torch.sum(h, dim=1)
        # print(h.shape)

        # run linear
        o = self.linear(h)
        # print(o.shape)

        return o, self.softmax(o)


class HELMoEncoder(nn.Module):
    """
    A Hierarchical LSTM + ELMo.
    Inputs: 
        x1: (batch_size, seq_len, 50),
        x2: (batch_size, seq_len, 50),
        x3: (batch_size, seq_len, 50),
    Outputs: (batch_size, C)
    """
    def __init__(self, I, H, C, L=1, bi=False, mlp=False, pre=False, 
                attentive=False, multiattentive=False, num_heads=4, 
                double_supervision=False, dropout=0.5):
        super(HELMoEncoder, self).__init__()
        self.bi = bi
        self.mlp = mlp
        self.pre = pre
        self.attentive = attentive
        self.multiattentive = multiattentive
        self.double_supervision = double_supervision

        self.I = I

        if not pre:
            self.gru = nn.GRU(self.I, hidden_size=H, num_layers=L, bidirectional=bi)
            self.elmo = Elmo(options_file, weight_file, 1, dropout=dropout)
        else:
            self.gru = nn.GRU(self.I, hidden_size=H, num_layers=L, bidirectional=bi)

        if mlp:
            self.mlp = nn.Linear(self.I * 3, H)

        D = H * 2 if bi else H
        self.linear = nn.Linear(D, C)
        self.softmax = nn.Softmax(dim=1)

        if self.multiattentive:
            self.sentences_attention = MultiHeadAttention(D, D, D, D, num_heads)
        elif self.attentive:
            self.sentences_attention = Attention(D)

        self.double_supervision = double_supervision
        if double_supervision:
            self.output_super = nn.Linear(self.I * 3, 4) # lower layer supervision


    def forward(self, x1, x2, x3):
        # run through ELMo ScalarMix
        if not self.pre:
            h1 = self.elmo(x1)['elmo_representations'][0]
            h2 = self.elmo(x2)['elmo_representations'][0]
            h3 = self.elmo(x3)['elmo_representations'][0]
            
            # sum dim=1
            h1 = torch.sum(h1, dim=1)
            h2 = torch.sum(h2, dim=1)
            h3 = torch.sum(h3, dim=1)
            super_h = torch.cat([h1, h2, h3], dim=1)
        else:
            h1 = x1
            h2 = x2
            h3 = x3
            # super_h = h1 + h2 + h3
            super_h = torch.cat([h1, h2, h3], dim=1)

        # print(h1.shape)
        # print(super_h.shape)

        # run through GRU - pre_elmo compatible
        if not self.mlp:
            hs = torch.stack([h1, h2, h3], dim=0)
            # print(hs.shape)
            hs, h = self.gru(hs)
            if self.bi:
                h = torch.cat((h[0], h[1]), dim=1)
            else:
                h = h.squeeze()

            if self.attentive:
                hs = hs.permute(1,0,2) #(batch, seq_len, num_directions * hidden_size)
                # print(hs.shape)
                if self.multiattentive:
                    h = torch.sum(self.sentences_attention(hs, hs, hs), dim=1)
                else:
                    h = self.sentences_attention(hs) #(batch, num_directions * hidden_size)
                
            # print(h.shape)

        # run through MLP - not pre_elmo compatible (works but 3072*3 dims?)
        else:
            # if self.pre:
            #     hs = h1 + h2 + h3
            # else:
            #     hs = torch.cat([h1, h2, h3], dim=1)
            hs = torch.cat([h1, h2, h3], dim=1)
            # print(hs.shape)
            h = F.relu(self.mlp(hs))
            # print(h.shape)

        # run linear
        if len(h.shape) == 1:
            h = h.unsqueeze(0)
        o = self.linear(h)
        # print(o.shape)

        # only pre_elmo compatible
        if self.double_supervision:
            additional_logits = self.output_super(super_h) # (batch_size, 4)
            return o, self.softmax(o), additional_logits, self.softmax(additional_logits)

        return o, self.softmax(o)

if __name__ == "__main__":
    x1 = torch.randint(0, 49, (1, 10, 50)).long() #(batch_size, seq_len, 50)
    x2 = torch.randint(0, 49, (1, 10, 50)).long() #(batch_size, seq_len, 50)
    x3 = torch.randint(0, 49, (1, 10, 50)).long() #(batch_size, seq_len, 50)
    model = HELMoEncoder(
        H=constant.hidden_dim, 
        L=constant.n_layers, 
        bi=constant.bidirec, 
        C=4, 
        mlp=constant.mlp, 
        pre=constant.use_elmo_pre,
        attentive=constant.attn,
        multiattentive=constant.multiattn,
        num_heads=constant.heads,
        double_supervision=constant.double_supervision,
    )    
    model(x1, x2, x3)
    # tt = MyTokenizer()
    # for X, x_len, y, ind, X_text in data_loaders_tr[0]:
    #     print(X_text)
    #     x = batch_to_ids([list(tt.tokenize(clean_sentence(d[0]+' '+d[1]+' '+d[2]))) for d in X_text])
    #     print(x)
    #     o = [elmo(x)]
    #     break