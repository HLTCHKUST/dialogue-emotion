from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from utils.data_reader import read_data ,get_data_for_bert
from utils import constant
from utils.utils import getMetrics
from models.transformer import Encoder
from models.lstm_model import HLstmModel
from models.common_layer import NoamOpt, Attention

import argparse
import collections
import logging
import json
import re
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.optim import Adam

from pytorch_pretrained_bert.tokenization import convert_to_unicode, BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def save_model(model, split):
    model_save_path = os.path.join(constant.save_path, "model_{}".format(split))
    torch.save(model.state_dict(), model_save_path)


def load_model(model, split):
    model_save_path = os.path.join(constant.save_path, "model_{}".format(split))
    state = torch.load(model_save_path, map_location=lambda storage, location: storage)
    model.load_state_dict(state)
    return model

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b, text_c, label):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, label_id):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_id = label_id
def read_examples(data, no_label = False):
    """Read a list of `InputExample`s from an input file."""
    examples = []
   
    if no_label:
        for id, sent in zip(*data):
            examples.append(
                    InputExample(unique_id=convert_to_unicode(str(id)),
                                    text_a=convert_to_unicode(sent[0]),
                                    text_b=convert_to_unicode(sent[1]), 
                                    text_c=convert_to_unicode(sent[2]),
                                    label = convert_to_unicode('others')))
    else:
        for id, sent, lab in zip(*data):
            examples.append(
                    InputExample(unique_id=convert_to_unicode(str(id)),
                                    text_a=convert_to_unicode(sent[0]),
                                    text_b=convert_to_unicode(sent[1]), 
                                    text_c=convert_to_unicode(sent[2]),
                                    label = convert_to_unicode(lab)))
    return examples

def _truncate_seq_pair(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        tokens.pop()


def convert_examples_to_features(examples, seq_length, tokenizer, hier=True):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    label_map = {"others":0, "happy":1, "sad":2, "angry":3}
    total_tokens = 0
    unk_tokens = 0
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    def _convert_one(ex_index,text, seq_length, tokenizer, total_tokens = 0, unk_tokens = 0):
        tokens_a = tokenizer.tokenize(text)
        _truncate_seq_pair(tokens_a, seq_length - 2)
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            if token == "[UNK]":
                unk_tokens+=1
            total_tokens+=1
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
        if hier:
            return tokens, input_ids, input_mask, input_type_ids, total_tokens, unk_tokens
        else:
            return tokens, input_ids, input_mask, input_type_ids
    if hier:
        for (ex_index, example) in enumerate(examples):
            tokens_a, input_ids_a, input_mask_a, input_type_ids_a, total_tokens, unk_tokens = _convert_one(ex_index, example.text_a, seq_length, tokenizer, total_tokens, unk_tokens)
            tokens_b, input_ids_b, input_mask_b, input_type_ids_b, total_tokens, unk_tokens = _convert_one(ex_index, example.text_b, seq_length, tokenizer, total_tokens, unk_tokens)
            tokens_c, input_ids_c, input_mask_c, input_type_ids_c, total_tokens, unk_tokens = _convert_one(ex_index, example.text_c, seq_length, tokenizer, total_tokens, unk_tokens)
            features.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    tokens=[tokens_a,tokens_b,tokens_c],
                    input_ids=[input_ids_a,input_ids_b,input_ids_c],
                    input_mask=[input_mask_a,input_mask_b,input_mask_c],
                    input_type_ids=[input_type_ids_a,input_type_ids_b,input_type_ids_c],
                    label_id = label_map[example.label]
                    ))
        print("============================================================")
        print('unkonw tokens percentage:{}'.format(unk_tokens/total_tokens))
        print("============================================================")
        return features
    else:
        for (ex_index, example) in enumerate(examples):
            tokens, input_ids, input_mask, input_type_ids = _convert_one(ex_index, example.text_a+example.text_b+example.text_c, seq_length, tokenizer)
            features.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids,
                    label_id = label_map[example.label]
                    ))
        return features


class HierBertModel(nn.Module):
    def __init__(self, label_num = 4, fix_bert = False, context_encoder=None, bilstm=True, dropout=0, double_supervision=False, emoji_vectors=None):
        super(HierBertModel, self).__init__()
        self.sentences_encoder = BertModel.from_pretrained('bert-base-cased')
        self.context_encoder = context_encoder
        self.bilstm = bilstm
        self.double_supervision = double_supervision
        self.emoji_emb = nn.Embedding.from_pretrained(torch.FloatTensor(emoji_vectors))
        self.emoji_dim = emoji_vectors.shape[1]
        if fix_bert:
            for param in self.sentences_encoder.parameters():
                param.requires_grad = False
        
        self.hidden_size = self.sentences_encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        if context_encoder =='lstm':
            self.hier_encoder = nn.LSTM(self.hidden_size, self.hidden_size,batch_first=True, bidirectional=bilstm)
        else:
            self.hier_encoder = Encoder(self.hidden_size+self.emoji_dim if (emoji_vectors is not None) else self.hidden_size , self.hidden_size, constant.hop, constant.heads, constant.depth, constant.depth,
                                    constant.filter, max_length=3, input_dropout=0, layer_dropout=0,
                                    attention_dropout=0, relu_dropout=0, use_mask=False, act=constant.act)
        self.classifer = nn.Linear(self.hidden_size, label_num)
        if double_supervision:
            self.super = nn.Linear((self.hidden_size+self.emoji_dim)*3 if (emoji_vectors is not None) else self.hidden_size*3, label_num)

        if (self.context_encoder =='lstm' and self.bilstm):
            self.classifer = nn.Linear(self.hidden_size*2, label_num)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sum_tensor=False, train=False, emoji_tokens = None, last_hidden=False):
        #encoded_layer:[batch_size, sequence_length, hidden_size]
        
        encoded_layer_a, pooled_output_a = self.sentences_encoder(input_ids[:,0], token_type_ids[:,0], attention_mask[:,0], output_all_encoded_layers=False)
        encoded_layer_b, pooled_output_b = self.sentences_encoder(input_ids[:,1], token_type_ids[:,1], attention_mask[:,1], output_all_encoded_layers=False)
        encoded_layer_c, pooled_output_c = self.sentences_encoder(input_ids[:,2], token_type_ids[:,2], attention_mask[:,2], output_all_encoded_layers=False)
        if sum_tensor:
            pooled_output_a = torch.sum(encoded_layer_a, dim=1) #[batch_size, hidden_size]
            pooled_output_b = torch.sum(encoded_layer_b, dim=1)
            pooled_output_c = torch.sum(encoded_layer_c, dim=1)
        if (emoji_tokens is not None):
            emoji_a = torch.sum(self.emoji_emb(emoji_tokens[:,0]), dim=1) #[batch_size, emoji_dim]
            emoji_b = torch.sum(self.emoji_emb(emoji_tokens[:,1]), dim=1)
            emoji_c = torch.sum(self.emoji_emb(emoji_tokens[:,2]), dim=1)
            #print('bert_output_dim:{}, emoji_dim:{}, emoji_tokens:{}'.format(pooled_output_a.size(), emoji_a.size(), emoji_tokens[0].size()))
            pooled_output_a = torch.cat((pooled_output_a,emoji_a),1)
            pooled_output_b = torch.cat((pooled_output_b,emoji_b),1)
            pooled_output_c = torch.cat((pooled_output_c,emoji_c),1)
        pooled_output_a = self.dropout(pooled_output_a)
        pooled_output_b = self.dropout(pooled_output_b)
        pooled_output_c = self.dropout(pooled_output_c)
        if self.double_supervision:
            additional_logits = self.super(torch.cat((pooled_output_a,pooled_output_b,pooled_output_c),dim=1))
        sum_pool = self.hier_encoder(torch.stack([pooled_output_a,pooled_output_b,pooled_output_c],dim=1))[0]
        sum_pool = self.dropout(sum_pool)
        if self.context_encoder=='lstm':
            sum_pool = sum_pool[:,-1,:]
            logits = self.classifer(sum_pool)
        else:
            if last_hidden:
                logits = self.classifer(sum_pool[:,-1])
            else:
                logits = self.classifer(torch.sum(sum_pool,dim=1)/sum_pool.size(1))
        if (self.double_supervision and train):
            return (logits, additional_logits)
        else:
            return logits

class FlatBertModel(nn.Module):
    def __init__(self, label_num = 4):
        super(FlatBertModel, self).__init__()
        self.sentences_encoder = BertModel.from_pretrained('bert-base-cased')
        self.hidden_size = self.sentences_encoder.config.hidden_size
        
        self.classifer = nn.Linear(self.hidden_size, label_num)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sum_tensor=False, criterion=None, label_ids=None):
        #encoded_layer:[batch_size, sequence_length, hidden_size]
        encoded_layer, pooled_output = self.sentences_encoder(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        if sum_tensor:
            sum_pool = torch.sum(encoded_layer, dim=1) #[batch_size, hidden_size]
        else:
            sum_pool = pooled_output
        logits = self.classifer(sum_pool)
        return logits


def predict_hier(model, criterion, loader, split=0):
    label2emotion = ["others","happy", "sad","angry"]
    model.eval()
    file = constant.save_path+"test_{}.txt".format(split)
    with open(file, 'w') as the_file:
        the_file.write("id\tturn1\tturn2\tturn3\tlabel\n")
        preds_dict = {}
        indices = []
        count = 0
        for X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text in loader:
            if x1_len is None:
                pred_prob = model(X_1, X_2, X_3)
            else:
                pred_prob = model(X_1, X_2, X_3, x1_len, x2_len, x3_len)

            preds = pred_prob[1].data.max(1)[1] # max func return (max, argmax)
            for idx, text, pred in zip(ind,X_text,preds):
                preds_dict[idx] = "{}\t{}\t{}\t{}\t{}\n".format(idx,text[0],text[1],text[2],label2emotion[pred.item()])
                indices.append(idx)
            
        sorted_indices = np.argsort(-np.array(indices))[::-1]
        for idx in range(len(sorted_indices)):
            the_file.write(preds_dict[idx])
    print("FILE {} SAVED".format(file))

def predict():
    args = constant.arg
    if not os.path.exists(constant.save_path):
        os.makedirs(constant.save_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    for seed in range(10):
        train, val, test_nolab, emoji_tokens, emoji_vectors= get_data_for_bert(seed=seed, emoji_dim=args.emoji_dim)
        train_emojis, val_emojis, test_emojis = emoji_tokens
        ids_test, sents_test, _ = test_nolab
        test_examples = read_examples((ids_test, sents_test), no_label=True)
        max_seq_length=40
        test_features = convert_examples_to_features(
            examples=test_examples, seq_length=max_seq_length, tokenizer=tokenizer, hier=args.hier)
        if args.hier:
            
            model = HierBertModel(context_encoder = args.context_encoder, dropout=args.dropout, double_supervision=args.double_supervision, emoji_vectors = emoji_vectors if args.emoji_emb else None)
            print(model)
        model = load_model(model, seed)
        model.cuda()
        #=====================test dataloader========================
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.input_type_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_emoji_tokens = torch.tensor([emojis for emojis in test_emojis], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_emoji_tokens)

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.batch_size)
        model.eval()
        all_logits = []
        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids, emoji_tokens = batch
            logits = model(input_ids, segment_ids, input_mask, args.sum_tensor, emoji_tokens= emoji_tokens if args.emoji_emb else None, last_hidden=args.last_hidden)
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits)
        pred = np.argmax(all_logits, axis=1)


        label2emotion = ["others","happy", "sad","angry"]
        file = constant.save_path+"test_{}.txt".format(seed)
        with open(file, 'w') as the_file:
            the_file.write("id\tturn1\tturn2\tturn3\tlabel\n")
            for idx, text, pred in zip(ids_test, sents_test, list(pred)):
                the_file.write("{}\t{}\t{}\t{}\t{}\n".format(idx,text[0],text[1],text[2],label2emotion[pred]))
        print("FILE {} SAVED".format(file))
def main():
    
    args = constant.arg
    if not os.path.exists(constant.save_path):
        os.makedirs(constant.save_path)
    #device = torch.device("cuda", 3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    f1_avg = []
    for seed in range(10):
        train, val, val_nolab , emoji_tokens, emoji_vectors = get_data_for_bert(seed=seed, emoji_dim=args.emoji_dim)
        train_emojis, val_emojis, test_emojis = emoji_tokens
        train_examples = read_examples(train)
        val_examples = read_examples(val)
        if args.hier:
            max_seq_length=40
        else:
            max_seq_length=100
        train_features = convert_examples_to_features(
            examples=train_examples, seq_length=max_seq_length, tokenizer=tokenizer, hier=args.hier)

        val_features = convert_examples_to_features(
            examples=val_examples, seq_length=max_seq_length, tokenizer=tokenizer, hier=args.hier)
        if args.hier:
            model = HierBertModel(context_encoder = args.context_encoder, dropout=args.dropout, double_supervision=args.double_supervision, emoji_vectors= emoji_vectors if args.emoji_emb else None)
        else:
            model = FlatBertModel()
        criterion = nn.CrossEntropyLoss()
        model.cuda()

        # Prepare optimizer
        if args.use_bertadam:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=5e-5,
                                warmup=0.02,
                                t_total=int(len(train_examples) / args.batch_size / 1 * 15))

        elif args.noam:
            optimizer = NoamOpt(
                constant.emb_dim,
                1,
                4000,
                torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )
        else:
            optimizer = Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=1e-3)
        
        #training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.batch_size)
        #=====================training dataloader========================
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.input_type_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_emoji_tokens = torch.tensor([emojis for emojis in train_emojis], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_emoji_tokens)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        
        #=====================val dataloader========================
        all_input_ids = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in val_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.input_type_ids for f in val_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in val_features], dtype=torch.long)
        all_emoji_tokens = torch.tensor([emojis for emojis in val_emojis], dtype=torch.long)
        val_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_emoji_tokens)

        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

        best_f1 = 0
        early_stop = 0
        for _ in trange(100, desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_steps = 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids, emoji_tokens = batch
                #print(input_ids.size())
                logits = model(input_ids, segment_ids, input_mask, args.sum_tensor, train=True, emoji_tokens=emoji_tokens if args.emoji_emb else None, last_hidden=args.last_hidden)
                #print(logits.size(), label_ids.size())
                if len(logits) ==2:
                    loss = (1-args.super_ratio)*criterion(logits[0],label_ids) + args.super_ratio*criterion(logits[1],label_ids)
                else:   
                    loss = criterion(logits,label_ids)

                loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1
                optimizer.step()
                model.zero_grad()
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(val_examples))
            logger.info("  Batch size = %d", args.batch_size)
            model.eval()
            all_logits = []
            all_labels = []
            for step, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids, emoji_tokens = batch
                logits = model(input_ids, segment_ids, input_mask, args.sum_tensor, emoji_tokens=emoji_tokens if args.emoji_emb else None, last_hidden=args.last_hidden)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                all_logits.append(logits)
                all_labels.append(label_ids)
            accuracy, microPrecision, microRecall, microF1 = getMetrics(np.concatenate(all_logits),np.concatenate(all_labels),verbose=True)
            if best_f1<microF1:
                best_f1 = microF1
                save_model(model,seed)
            else:
                early_stop+=1
                if early_stop>5:
                    break
        print('EXPERIMENT:{}, best_f1:{}'.format(seed,best_f1))
        f1_avg.append(best_f1)
    
    file_summary = constant.save_path+"summary.txt"
    with open(file_summary, 'w') as the_file:
        header = "\t".join(["SPLIT_{}".format(i) for i, _ in enumerate(f1_avg)])
        the_file.write(header+"\tAVG\n")
        ris = "\t".join(["{:.4f}".format(e) for i, e in enumerate(f1_avg)])
        the_file.write(ris+"\t{:.4f}\n".format(np.mean(f1_avg)))
   


if __name__ == "__main__":
    if constant.test:
        predict()
    else:
        main()