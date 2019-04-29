import argparse
import random
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, default=100)
parser.add_argument("--emb_dim", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--save_path", type=str, default="save/")
parser.add_argument("--load_model_path", type=str, default="save/") # load trained model
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--pretrain_emb", action="store_true")
parser.add_argument("--mlp", action="store_true")
parser.add_argument("--elmo", action="store_true")
parser.add_argument("--use_elmo_pre", action="store_true")
parser.add_argument("--test", action="store_true")
# parser.add_argument("--dev_with_label", action="store_true")
# parser.add_argument("--include_test", action="store_true") # include test vocab and merge train and dev
parser.add_argument("--model", type=str, default="UTRS") # UTRS, LSTM
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", action="store_true")
parser.add_argument("--noam", action="store_true")
parser.add_argument("--mask", action="store_true")
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--num_split", type=int, default=10)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--patient", type=int, default=10)
parser.add_argument("--pretrain_list", type=str, default="")
parser.add_argument("--fix_pretrain", action="store_true")

parser.add_argument("--save_prediction_path", type=str, default="save/")
parser.add_argument("--save_confidence_path", type=str, default="save/")

## Double Supervision
parser.add_argument("--super_ratio", type=float, default=0.3)
parser.add_argument("--double_supervision", action="store_true")

## BERT
parser.add_argument("--hier", action="store_true")
parser.add_argument("--use_bertadam", action="store_true")
parser.add_argument("--sum_tensor", action="store_true")
parser.add_argument("--context_encoder", type=str, default='tras')
parser.add_argument("--dropout", type=float, default=0)
parser.add_argument("--emoji_emb", action="store_true")
parser.add_argument("--emoji_dim", type=int, default=100)
parser.add_argument("--last_hidden", action="store_true")
## LSTM
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--bidirec", action="store_true")
parser.add_argument("--drop", type=float, default=0)
parser.add_argument("--attn", action="store_true")
parser.add_argument("--multiattn", action="store_true")
parser.add_argument("--context", action="store_true")
parser.add_argument("--pool", type=str, default='')

# Pooling
parser.add_argument("--pool_kernel", type=int, default=3)
parser.add_argument("--pool_stride", type=int, default=2)

## Transformer 
parser.add_argument("--hop", type=int, default=6)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--depth_val", type=int, default=40)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)

parser.add_argument("--input_dropout", type=float, default=0)
parser.add_argument("--layer_dropout", type=float, default=0)
parser.add_argument("--attention_dropout", type=float, default=0)
parser.add_argument("--relu_dropout", type=float, default=0)

# Evaluation
parser.add_argument("--pred_file_path", type=str, default="")
parser.add_argument("--ground_file_path", type=str, default="")

# Predict
parser.add_argument("--emoji_filter", action="store_true")

## Predict with feature-based classfiers
parser.add_argument("--C", type=float, default=1)
parser.add_argument("--n_estimators", type=int, default=600)
parser.add_argument("--max_depth", type=int, default=6)
parser.add_argument("--min_child_weight", type=int, default=1)
parser.add_argument("--classifier", type=str, default="LR")
parser.add_argument("--features", action='append', default=[])
parser.add_argument("--pred_score", action="store_true")

# Ensemble with SVM (Evaluation time only)
parser.add_argument("--use_svm", action="store_true")

# Ensemble with XGBoost (Evaluation time only)
parser.add_argument("--use_xgb", action="store_true")
parser.add_argument("--evaluate_type", type=str)

# Voting
parser.add_argument('--voting-dir-list', nargs='+', type=str)
parser.add_argument("--voting-threshold", type=float, default=0)

# Preprocessing
parser.add_argument("--extra_prep", action="store_true")

arg = parser.parse_args()
print(arg)
model = arg.model

evaluate = False

# Hyperparameters
hidden_dim = arg.hidden_dim
emb_dim = arg.emb_dim
batch_size = arg.batch_size
lr = arg.lr
mlp = arg.mlp
seed = arg.seed
num_split = arg.num_split
max_epochs = arg.max_epochs
attn = arg.attn

USE_CUDA = arg.cuda
UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
CLS_idx = 4

### pretrained embeddings
# emb_file = "vectors/glove/glove.6B.{}d.txt".format(str(emb_dim))
emb_map = {"glove840B~300" : "vectors/glove/glove.840B.300d.txt", "emoji~300": "vectors/emoji/emoji_embeddings_300d.txt", "emo2vec~100": "vectors/emo2vec/emo2vec.txt"}

emb_file = arg.pretrain_list
emb_file_list = []
if emb_file != "":
    for emb in emb_file.split(","):
        emb_file_list.append(emb_map[emb] + "~" + emb.split("~")[1])

emb_dim_list = []
pretrained = arg.pretrain_emb
if(pretrained):
    emb_dim = arg.emb_dim

fix_pretrain = arg.fix_pretrain
super_ratio = arg.super_ratio
double_supervision = arg.double_supervision

save_path = arg.save_path
load_model_path = arg.load_model_path
test = arg.test
elmo = arg.elmo
use_elmo_pre = arg.use_elmo_pre
# dev_with_label = arg.dev_with_label
include_test = True

### lstm
n_layers = arg.n_layers
bidirec = arg.bidirec
drop = arg.drop
attn = arg.attn
multiattn = arg.multiattn

# Pooling
pool_kernel = arg.pool_kernel
pool_stride = arg.pool_stride

### transformer 
hop = arg.hop
heads = arg.heads
depth = arg.depth
depth_val = arg.depth_val

input_dropout = arg.input_dropout
layer_dropout = arg.layer_dropout
attention_dropout = arg.attention_dropout
relu_dropout = arg.relu_dropout

filter = arg.filter

label_smoothing = arg.label_smoothing
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight
mask = arg.mask
patient = arg.patient
attn = arg.attn
context = arg.context
pool = arg.pool
emoji = arg.emoji_emb

## Meta-learn
meta_lr = 1.0

## Seed
np.random.seed(seed)
torch.manual_seed(seed)
if USE_CUDA:
    torch.cuda.manual_seed_all(seed)
device = arg.device

## eval
pred_file_path = arg.pred_file_path
ground_file_path = arg.ground_file_path

# Predict
emoji_filter = arg.emoji_filter

## predict_classifier
C = arg.C
n_estimators = arg.n_estimators
max_depth = arg.max_depth
min_child_weight = arg.min_child_weight
classifier = arg.classifier
features = arg.features
pred_score = arg.pred_score

# Ensemble with SVM (Evaluation time only)
use_svm = arg.use_svm

# Ensemble with XGBoost (Evaluation time only)
use_xgb = arg.use_xgb

evaluate_type = arg.evaluate_type

# voting
voting_dir_list = arg.voting_dir_list
voting_threshold = arg.voting_threshold

# preprocessing
extra_prep = arg.extra_prep

save_prediction_path = arg.save_prediction_path
save_confidence_path = arg.save_confidence_path

evaluate=False
