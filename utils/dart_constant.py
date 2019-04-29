import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description="DARTS")
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.1,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.1,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.1,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--device", type=int, default=0)
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')
parser.add_argument("--noam", action="store_true")

parser.add_argument("--num_split", type=int, default=10)
parser.add_argument("--pretrain_emb", action="store_true")
parser.add_argument("--elmo", action="store_true")
arg = parser.parse_args()

# Hyperparameters
emb_dim = arg.emb_dim
hidden_dim = arg.nhid
hidden_dim_last = arg.nhidlast
lr = arg.lr
clip = arg.clip
max_epochs = arg.max_epochs
batch_size = arg.batch_size
bptt = arg.bptt
dropout = arg.dropout
dropouth = arg.dropouth
dropoutx = arg.dropoutx
dropouti = arg.dropouti
dropoute = arg.dropoute
seed = arg.seed
nonmono = arg.nonmono
cuda = arg.cuda
device = arg.device
log_interval = arg.log_interval
save = arg.save
alpha = arg.alpha
beta = arg.beta
wdecay = arg.wdecay
continue_train = arg.continue_train
small_batch_size = arg.small_batch_size
max_seq_len_delta = arg.max_seq_len_delta
single_gpu = arg.single_gpu
gpu = arg.gpu
unrolled = arg.unrolled
arch_wdecay = arg.arch_wdecay
arch_lr = arg.arch_lr
noam = arg.noam

num_split = arg.num_split
pretrain_emb = arg.pretrain_emb
elmo = arg.elmo

emb_file = "vectors/glove/glove.840B.300d.txt"
preptrained = arg.pretrain_emb
if(preptrained): 
    emb_dim = 300

USE_CUDA = arg.cuda
UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3

## Seed
np.random.seed(seed)
torch.manual_seed(seed)
if USE_CUDA:
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.enabled = True

# parser = argparse.ArgumentParser()
# parser.add_argument("--hidden_dim", type=int, default=100)
# parser.add_argument("--emb_dim", type=int, default=100)
# parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--lr", type=float, default=0.001)
# parser.add_argument("--save_path", type=str, default="save/")
# parser.add_argument("--cuda", action="store_true")
# parser.add_argument("--pretrain_emb", action="store_true")
# parser.add_argument("--elmo", action="store_true")
# parser.add_argument("--test", action="store_true")
# parser.add_argument("--model", type=str, default="UTRS") # UTRS, LSTM
# parser.add_argument("--weight_sharing", action="store_true")
# parser.add_argument("--label_smoothing", action="store_true")
# parser.add_argument("--noam", action="store_true")
# parser.add_argument("--universal", action="store_true")
# parser.add_argument("--act", action="store_true")
# parser.add_argument("--act_loss_weight", type=float, default=0.001)
# parser.add_argument("--seed", type=int, default=1234)

# parser.add_argument("--num_split", type=int, default=10)
# parser.add_argument("--max_epochs", type=int, default=100)
# ## bert
# parser.add_argument("--hier", action="store_true")
# parser.add_argument("--use_bertadam", action="store_true")
# ## lstm
# parser.add_argument("--n_layers", type=int, default=1)
# parser.add_argument("--bidirec", action="store_true")
# parser.add_argument("--drop", type=float, default=0)

# ## transformer 
# parser.add_argument("--hop", type=int, default=6)
# parser.add_argument("--heads", type=int, default=4)
# parser.add_argument("--depth", type=int, default=40)
# parser.add_argument("--filter", type=int, default=50)

# arg = parser.parse_args()
# print(arg)
# model = arg.model


# # Hyperparameters
# hidden_dim= arg.hidden_dim
# emb_dim= arg.emb_dim
# batch_size= arg.batch_size
# lr=arg.lr
# seed = arg.seed
# num_split = arg.num_split
# max_epochs = arg.max_epochs


# # emb_file = "vectors/glove/glove.6B.{}d.txt".format(str(emb_dim))
# emb_file = "vectors/glove/glove.840B.300d.txt"
# preptrained = arg.pretrain_emb
# if(preptrained): 
#     emb_dim = 300

# save_path = arg.save_path
# test = arg.test
# elmo = arg.elmo

# ### lstm
# n_layers = arg.n_layers
# bidirec = arg.bidirec
# drop = arg.drop

# ### transformer 
# hop = arg.hop
# heads = arg.heads
# depth = arg.depth
# filter = arg.filter


# label_smoothing = arg.label_smoothing
# weight_sharing = arg.weight_sharing
# noam = arg.noam
# universal = arg.universal
# act = arg.act
# act_loss_weight = arg.act_loss_weight


# ## Meta-learn
# meta_lr = 1.0

