import os
import re

import nltk
import dill as pickle
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from operator import itemgetter

import torch
import torch.utils.data as data
from scipy.stats import itemfreq
from sklearn.utils import shuffle
from allennlp.modules.elmo import batch_to_ids as b2id

from utils import constant
from utils.to_emoji import text_to_emoji
from utils.utils import ImbalancedDatasetSampler

from torch.utils.data import ConcatDataset

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

from allennlp.commands.elmo import ElmoEmbedder

import string

def clean_sentence(sent):
    # remove new line token
    sent = re.sub("&amp;", " ", sent)
    sent = re.sub("&apos;", "'", sent)
    # use special number tokenS
    sent = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", sent)
    # remove repetition
    sent = re.sub(r"([!?.]){2,}", r"\1", sent)
    sent = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2", sent)
    sent = re.sub(r"\.", r" . ", sent)
    # remove_nonalphanumeric:
    # sent = re.sub(r"([^\s\w\@!?<>]|_)+", "", sent)
    # use_user_special_token:
    sent = re.sub(r"@\w+", "<user>", sent)
    return sent

class MyTokenizer():
    def __init__(self):
        self.tt = nltk.tokenize.TweetTokenizer()

    def tokenize(self, sentence):
        tok = []
        if constant.extra_prep:
            special_cases = ["'m", "'s", "'ve", "n't", "'re", "'d", "'ll"]    
        else:
            special_cases = ["'s", "'ve", "n't", "'re", "'d", "'ll"]
        # new = self.tt.tokenize(clean_sentence(sentence.lower()))
        # old = self.tt.tokenize(sentence.lower())
        # if len(set(new)) > len(set(old)):
        # print(new, old)
        for word in self.tt.tokenize(clean_sentence(sentence.lower())):
            flag = False
            if(word in text_to_emoji):
                word = text_to_emoji[word]
            for case in special_cases:
                if case not in word:
                    continue
                idx = word.find(case)
                tok.append(word[:idx])
                tok.append(word[idx:])
                flag = True
                break
            if not flag:
                tok.append(word)
        return tok


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {constant.UNK_idx: "UNK", constant.PAD_idx: "PAD", constant.EOS_idx: "EOS", constant.SOS_idx: "SOS",  constant.CLS_idx: "CLS"} 
        self.n_words = 5 # Count default tokens
        self.tt = MyTokenizer()

    def tokenize(self, sentence):
        tok = []
        # new = self.tt.tokenize(clean_sentence(sentence.lower()))
        # old = self.tt.tokenize(sentence.lower())
        # if len(set(new)) > len(set(old)):
        #     print(new, old)
        for word in self.tt.tokenize(clean_sentence(sentence.lower())):
            if(word in text_to_emoji):
                word = text_to_emoji[word]
            tok.append(word)
        return tok
        
    def index_words(self, sentence):
        for word in self.tokenize(sentence):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab, hier=False, elmo=False, elmo_pre=None, deepmoji=False):
        self.id, self.X, self.y = data
        self.emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

        if(self.y is None):
            self.y = None
        else:
            self.y = np.array(list(map(lambda label: self.emotion2label[label], self.y)))
        self.vocab = vocab
        self.num_total_seqs = len(self.X)
        self.tt = MyTokenizer()

        with open(VOCAB_PATH, 'r') as f:
            deepmoji_vocab = json.load(f)
        self.deepmoji_tt = SentenceTokenizer(deepmoji_vocab, 100)

        self.hier = hier
        self.elmo = elmo
        self.elmo_pre = elmo_pre # pre-extracted elmo embeddings
        self.deepmoji = deepmoji

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ind = self.id[index]
        X_text = self.X[index]
        if(self.y is None): y = None
        else: y = self.y[index]
        if(self.hier): 
            if self.elmo_pre is not None:
                f = lambda l, d: itemgetter(*l)(d) # get Tuple(values) with List[keys]
                X_1,X_2,X_3 = self.X[index][0], self.X[index][1], self.X[index][2]
                return (*f([X_1.lower(),X_2.lower(),X_3.lower()], self.elmo_pre), y, ind, X_text)
            X_1,X_2,X_3 = self.preprocess(self.X[index])
            return X_1, X_2, X_3, y, ind, X_text
        else: 
            X = self.preprocess(self.X[index])
            return X, y, ind, X_text

    def __len__(self):
        return self.num_total_seqs

    def vectorize(self, sentence):
        sequence = []

        for word in self.tt.tokenize(clean_sentence(sentence)):
            if(word in text_to_emoji):
                word = text_to_emoji[word]

            # word = word.translate(None, string.punctuation)
            if constant.extra_prep:
                table = str.maketrans({key: None for key in string.punctuation})
                word = word.translate(table)
                if len(word) == 0:
                    continue

                # the following code maybe not useful at all
                old_word = word
                if word not in constant.gen_vocabs:
                    word = word.lower()
                if word not in constant.gen_vocabs:
                    word = word[0].upper() + word[1:]
                if word not in constant.gen_vocabs:
                    word = word.upper()
                if old_word not in constant.gen_vocabs:
                    if word in constant.gen_vocabs:
                        print(">", old_word, word)
                if word not in constant.gen_vocabs:
                    word = old_word

            if word in self.vocab.word2index:
                sequence.append(self.vocab.word2index[word]) 
            else:
                sequence.append(constant.UNK_idx)
        return sequence

    def preprocess(self, arr):
        """Converts words to ids."""
        t1 = 'CLS ' + arr[0].lower()
        t2 = 'CLS ' + arr[1].lower()
        t3 = 'CLS ' + arr[2].lower()

        # print("preprocess deepmoji=", self.deepmoji)

        if self.elmo:
            t1 = self.tt.tokenize(clean_sentence(t1))
            t2 = self.tt.tokenize(clean_sentence(t2))
            t3 = self.tt.tokenize(clean_sentence(t3))

            if self.hier:
                return t1, t2, t3
            else:
                return np.concatenate((t1, t2, t3))
        elif self.deepmoji:
            t1, _, _ = self.deepmoji_tt.tokenize_sentences([t1]) #vectorize
            t2, _, _ = self.deepmoji_tt.tokenize_sentences([t2])
            t3, _, _ = self.deepmoji_tt.tokenize_sentences([t3])

            t1 = np.trim_zeros(t1.astype(np.int32)[0])
            t2 = np.trim_zeros(t2.astype(np.int32)[0])
            t3 = np.trim_zeros(t3.astype(np.int32)[0])

            if self.hier:
                return torch.LongTensor(t1),torch.LongTensor(t2),torch.LongTensor(t3)
            else:
                return torch.LongTensor(t1+t2+t3)
        else:
            t1 = self.vectorize(t1)
            t2 = self.vectorize(t2)
            t3 = self.vectorize(t3)
            
            if self.hier:
                return torch.LongTensor(t1),torch.LongTensor(t2),torch.LongTensor(t3)
            else:
                return torch.LongTensor(t1+t2+t3)

def collate(hier=False, elmo=False, use_elmo_pre=False, deepmoji=False):
    def collate_fn(data):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths 

        data.sort(key=lambda x: len(x[0]), reverse=True)
        X, y, ind, X_text = zip(*data)
        if elmo:
            X = b2id(X)
            x_len = None
        else:
            X, x_len = merge(X)

        X = torch.LongTensor(X)

        if(y[0] is None): pass
        else: y = torch.LongTensor(y)

        if constant.USE_CUDA:
            device = torch.device('cuda:{}'.format(constant.device)) 
            X = X.to(device)
            if(y[0] is None): pass
            else: y = y.to(device)

        return X, x_len, y, ind, X_text
    
    def collate_fn_hier(data):
        def merge(sequences):
            # for seq in sequences:
            #     print(">", seq)
                # print(len(sequences))
            lengths = [len(seq) for seq in sequences]
            # print(lengths)
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                # print(seq[:end].size(), padded_seqs[i].size())
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths 


        if use_elmo_pre:
            X_1, X_2, X_3, y, ind, X_text = zip(*data)

            X_1 = torch.FloatTensor(X_1)
            X_2 = torch.FloatTensor(X_2)
            X_3 = torch.FloatTensor(X_3)

            if(y[0] is None): pass
            else: y = torch.LongTensor(y)
                
            if constant.USE_CUDA:
                device = torch.device('cuda:{}'.format(constant.device)) 
                X_1 = X_1.to(device)
                X_2 = X_2.to(device)
                X_3 = X_3.to(device)

                if(y[0] is None): pass
                else: y = y.to(device)

            return X_1, X_2, X_3, None, None, None, y, ind, X_text
   
        data.sort(key=lambda x: len(x[0]), reverse=True)
        X_1, X_2, X_3, y, ind, X_text = zip(*data)

        if elmo:
            X_1, X_2, X_3 = b2id(X_1), b2id(X_2), b2id(X_3)
            x1_len, x2_len, x3_len = None, None, None
        else:
            X_1, x1_len = merge(X_1)
            X_2, x2_len = merge(X_2)
            X_3, x3_len = merge(X_3)

        X_1 = torch.LongTensor(X_1)
        X_2 = torch.LongTensor(X_2)
        X_3 = torch.LongTensor(X_3)

        if(y[0] is None): pass
        else: y = torch.LongTensor(y)

        if constant.USE_CUDA:
            device = torch.device('cuda:{}'.format(constant.device)) 
            X_1 = X_1.to(device)
            X_2 = X_2.to(device)
            X_3 = X_3.to(device)

            if(y[0] is None): pass
            else: y = y.to(device)

        return X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text

    # Return inner function with closure
    return collate_fn_hier if hier else collate_fn

def generate_val_set(ids, sents, lab):
    train_ids, train_sent, train_label = [], [], []
    val_ids, val_sent, val_label  = [], [], []
    cnt_angry, cnt_happy ,cnt_others ,cnt_sad = 0,0,0,0

    for (i,s,l) in zip(ids, sents, lab):
        if(l == "angry" and cnt_angry < 50): 
            cnt_angry +=1 
            val_ids.append(i) 
            val_sent.append(s) 
            val_label.append(l)
        elif(l == "happy" and cnt_happy < 50): 
            cnt_happy +=1 
            val_ids.append(i) 
            val_sent.append(s) 
            val_label.append(l)
        elif(l == "others" and cnt_others < 1000): 
            cnt_others +=1 
            val_ids.append(i) 
            val_sent.append(s) 
            val_label.append(l)
        elif(l == "sad" and cnt_sad < 50): 
            cnt_sad += 1
            val_ids.append(i) 
            val_sent.append(s) 
            val_label.append(l)
        else:
            train_ids.append(i) 
            train_sent.append(s) 
            train_label.append(l)
    return (train_ids,train_sent,train_label), (val_ids,val_sent,val_label)

def generate_data_set(ids, sents, lab):
    data_ids, data_sent, data_label = [], [], []
    
    for (i,s,l) in zip(ids, sents, lab):
        data_ids.append(i) 
        data_sent.append(s) 
        data_label.append(l)
    return (data_ids,data_sent,data_label)


def read_data(is_shuffle=False, random_state=0, dev_with_label=False, include_test=False):
    # print("Reading Dataset, shuffle:", is_shuffle, ", random state:", random_state)
    ## training
    df = pd.read_csv('data/train.txt', delimiter='\t')
    if is_shuffle:
        df = shuffle(df, random_state=random_state)

    sents = df[['turn1', 'turn2', 'turn3']].values
    lab = df['label'].values
    ids = df['id']

    if constant.evaluate:
        print("read data EVALUATION")
        df_dev = pd.read_csv('data/dev.txt', delimiter='\t')
        sents_dev = df_dev[['turn1', 'turn2', 'turn3']].values
        lab_dev = df_dev['label'].values
        ids_dev = df_dev['id']
        
        train = generate_data_set(ids, sents, lab)
        val = generate_data_set(ids_dev, sents_dev, lab_dev)
    else:
        # merge train and dev
        if include_test:
            filepaths = ['data/train.txt', 'data/dev.txt']
            df = (pd.read_csv(f, delimiter='\t') for f in filepaths)
            df = pd.concat(df)
            
            if is_shuffle:
                df = shuffle(df, random_state=random_state)

            sents = df[['turn1', 'turn2', 'turn3']].values
            lab = df['label'].values
            ids = df['id']
            
            train, val = generate_val_set(ids, sents, lab)
        else:
            if dev_with_label:
                df_dev = pd.read_csv('data/dev.txt', delimiter='\t')
                sents_dev = df_dev[['turn1', 'turn2', 'turn3']].values
                lab_dev = df_dev['label'].values
                ids_dev = df_dev['id']
                
                train = generate_data_set(ids, sents, lab)
                val = generate_data_set(ids_dev, sents_dev, lab_dev)
            else:    
                train, val = generate_val_set(ids, sents, lab)

    ## DEV with no label
    df_dev = pd.read_csv('data/devwithoutlabels.txt', delimiter='\t')
    sents_dev = df_dev[['turn1', 'turn2', 'turn3']].values
    ids_dev = df_dev['id']

    ## TEST with no label
    df_test = pd.read_csv('data/test.txt', delimiter='\t')
    sents_test = df_test[['turn1', 'turn2', 'turn3']].values
    ids_test = df_test['id']

    # print("Frequency Original Label")
    # for it in itemfreq(lab):
    #     print(it[0],":",it[1])

    # print("Frequency Training Label")
    # for it in itemfreq(train[2]):
    #     print(it[0],":",it[1])

    # print("Frequency Validation Label")
    # for it in itemfreq(val[2]):
    #     print(it[0],":",it[1])

    # return train, val, (ids_dev, sents_dev, None)
    return train, val, (ids_test, sents_test, None)


def get_data_for_bert(seed, emoji_dim, max_emoji_num = 10):
    tt = MyTokenizer()
    emoji_file = 'vectors/emoji/emoji_embeddings_'+str(emoji_dim)+'d.txt'
    no_emoji = [float(0.0)]*emoji_dim
    emoji2index = {'no_emoji':0}
    index2emoji = {0:'no_emoji'}
    emoji_vec = [no_emoji,]
    with open(emoji_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            index2emoji[i+1] = line[0]
            emoji2index[line[0]] = i+1
            emoji_vec.append([float(x) for x in line[1:]])
    

    train, val, val_nolab = read_data(is_shuffle=True, random_state=seed, dev_with_label=constant.dev_with_label, include_test=constant.include_test)
    train_ids,train_sent,train_label = train
    val_ids,val_sent,val_label = val
    test_ids,test_sent,_ = val_nolab
    new_train_sent = []
    train_emojis = []
    new_val_sent = []
    val_emojis = []
    new_test_sent = []
    test_emojis = []
    for sent in train_sent:
        emojis = [[],[],[]]
        words = tt.tokenize(sent[0])
        for word in words:
            if word in emoji2index:
                emojis[0].append(emoji2index[word])
        if len(emojis[0])>max_emoji_num:
            emojis[0] = emojis[0][:max_emoji_num]
        else:
            emojis[0] = emojis[0] + [0] * (max_emoji_num - len(emojis[0]))  #pad emoji tokens
        sent[0] = ' '.join(words)

        words = tt.tokenize(sent[1])
        for word in words:
            if word in emoji2index:
                emojis[1].append(emoji2index[word])
        if len(emojis[1])>max_emoji_num:
            emojis[1] = emojis[1][:max_emoji_num]
        else:
            emojis[1] = emojis[1] + [0] * (max_emoji_num - len(emojis[1]))  #pad emoji tokens
        sent[1] = ' '.join(words)

        words = tt.tokenize(sent[2])
        for word in words:
            if word in emoji2index:
                emojis[2].append(emoji2index[word])
        if len(emojis[2])>max_emoji_num:
            emojis[2] = emojis[2][:max_emoji_num]
        else:
            emojis[2] = emojis[2] + [0] * (max_emoji_num - len(emojis[2]))  #pad emoji tokens
        sent[2] = ' '.join(words)
        new_train_sent.append(sent)
        train_emojis.append(emojis)
    for sent in val_sent:
        emojis = [[],[],[]]
        words = tt.tokenize(sent[0])
        for word in words:
            if word in emoji2index:
                emojis[0].append(emoji2index[word])
        if len(emojis[0])>max_emoji_num:
            emojis[0] = emojis[0][:max_emoji_num]
        else:
            emojis[0] = emojis[0] + [0] * (max_emoji_num - len(emojis[0]))  #pad emoji tokens
        sent[0] = ' '.join(words)

        words = tt.tokenize(sent[1])
        for word in words:
            if word in emoji2index:
                emojis[1].append(emoji2index[word])
        if len(emojis[1])>max_emoji_num:
            emojis[1] = emojis[1][:max_emoji_num]
        else:
            emojis[1] = emojis[1] + [0] * (max_emoji_num - len(emojis[1]))  #pad emoji tokens
        sent[1] = ' '.join(words)

        words = tt.tokenize(sent[2])
        for word in words:
            if word in emoji2index:
                emojis[2].append(emoji2index[word])
        if len(emojis[2])>max_emoji_num:
            emojis[2] = emojis[2][:max_emoji_num]
        else:
            emojis[2] = emojis[2] + [0] * (max_emoji_num - len(emojis[2]))  #pad emoji tokens
        sent[2] = ' '.join(words)
        new_val_sent.append(sent)
        val_emojis.append(emojis)
    for sent in test_sent:
        emojis = [[],[],[]]
        words = tt.tokenize(sent[0])
        for word in words:
            if word in emoji2index:
                emojis[0].append(emoji2index[word])
        if len(emojis[0])>max_emoji_num:
            emojis[0] = emojis[0][:max_emoji_num]
        else:
            emojis[0] = emojis[0] + [0] * (max_emoji_num - len(emojis[0]))  #pad emoji tokens
        sent[0] = ' '.join(words)

        words = tt.tokenize(sent[1])
        for word in words:
            if word in emoji2index:
                emojis[1].append(emoji2index[word])
        if len(emojis[1])>max_emoji_num:
            emojis[1] = emojis[1][:max_emoji_num]
        else:
            emojis[1] = emojis[1] + [0] * (max_emoji_num - len(emojis[1]))  #pad emoji tokens
        sent[1] = ' '.join(words)

        words = tt.tokenize(sent[2])
        for word in words:
            if word in emoji2index:
                emojis[2].append(emoji2index[word])
        if len(emojis[2])>max_emoji_num:
            emojis[2] = emojis[2][:max_emoji_num]
        else:
            emojis[2] = emojis[2] + [0] * (max_emoji_num - len(emojis[2]))  #pad emoji tokens
        sent[2] = ' '.join(words)
        new_test_sent.append(sent)
        test_emojis.append(emojis)
    return (train_ids,train_sent,train_label), (val_ids,val_sent,val_label), (test_ids,test_sent,None),(train_emojis, val_emojis, test_emojis), np.array(emoji_vec)

def generate_vocab(deepmoji=False, include_test=False):
    # print("Generate vocab")
    ## training
    df = pd.read_csv('data/train.txt', delimiter='\t')
    sents = df[['turn1', 'turn2', 'turn3']].values

    df_dev = pd.read_csv('data/dev.txt', delimiter='\t')
    sents_dev = df_dev[['turn1', 'turn2', 'turn3']].values

    ## DEV with no label
    df_dev_nolabel = pd.read_csv('data/devwithoutlabels.txt', delimiter='\t')
    sents_dev_nolabel = df_dev_nolabel[['turn1', 'turn2', 'turn3']].values

    ## TEST with no label
    df_test_nolabel = pd.read_csv('data/test.txt', delimiter='\t')
    sents_test_nolabel = df_test_nolabel[['turn1', 'turn2', 'turn3']].values

    vocab = Lang()
    for s in sents:
        vocab.index_words(s[0]) ## turn 1 
        vocab.index_words(s[1]) ## turn 2
        vocab.index_words(s[2]) ## turn 3
    # print("Vocab size no dev:",vocab.n_words)

    for s in sents_dev:
        vocab.index_words(s[0]) ## turn 1 
        vocab.index_words(s[1]) ## turn 2
        vocab.index_words(s[2]) ## turn 3
    # print("Vocab size with dev:",vocab.n_words)

    if include_test:
        # print("include_test")
        # for s in sents_dev_nolabel:
        #     vocab.index_words(s[0]) ## turn 1 
        #     vocab.index_words(s[1]) ## turn 2
        #     vocab.index_words(s[2]) ## turn 3
        for s in sents_test_nolabel:
            vocab.index_words(s[0]) ## turn 1 
            vocab.index_words(s[1]) ## turn 2
            vocab.index_words(s[2]) ## turn 3
        # print("Vocab size with test:",vocab.n_words)
    return vocab

def extract_elmo(emoji=False):

    def save_pkl(data, path):
        with open(path,'wb') as f:
            pickle.dump(data, f)

    def load_emoji_vec(dim=300):
        emoji_file = 'vectors/emoji/emoji_embeddings_'+str(dim)+'d.txt'
        no_emoji = [float(0.0)]*dim
        emoji2index = {'no_emoji':0}
        index2emoji = {0:'no_emoji'}
        emoji_vec = [no_emoji,]
        with open(emoji_file, encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip().split()
                index2emoji[i+1] = line[0]
                emoji2index[line[0]] = i+1
                emoji_vec.append([float(x) for x in line[1:]])

        emoji_vec = np.array(emoji_vec)
        print(emoji_vec.shape)
        return emoji_vec, emoji2index, index2emoji

    def find_emoji_vec(sent, emoji_vec, emoji2index):
        vec = emoji_vec[0]
        for emoji, idx in emoji2index.items():
            if emoji in sent:
                vec += emoji_vec[idx]
        return np.array(vec)

    def cat_emoji_vec(k, v, emoji_vec, emoji2index):
        vec = find_emoji_vec(k, emoji_vec, emoji2index)
        return np.concatenate((v, vec))

    save_path = 'data/elmo.pkl'
    
    if emoji:
        save_path = 'data/elmo_emoji.pkl'

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            elmos = pickle.load(f)
        return elmos
    elif emoji:
        with open('data/elmo.pkl', 'rb') as f:
            elmos = pickle.load(f)
        emoji_vec, emoji2index, _ = load_emoji_vec()
        elmos = {k: cat_emoji_vec(k, v, emoji_vec, emoji2index) for k, v in tqdm(elmos.items())}
        save_pkl(elmos, save_path)
        return elmos
    else:
        def tokenizer(ds):
            for d in ds:
                yield nltk.word_tokenize(d)
                
        def transform(tensor):
            """[T, L, W, D] => [T, W, L, D] => [T, W, L*D] => [T, L*D]"""
            L, W, D = tensor.shape
            return tensor.transpose(1, 0, 2).reshape(W, L*D).mean(axis=0)

        train, val, dev = read_data()
        dialogs = train[1] + val[1] + dev[1].tolist()
        us = {}
        for u in tqdm(np.array(dialogs).flat):
            us[u.lower()] = True

        elmo = ElmoEmbedder(cuda_device=0)
        us_elmo = {k: transform(elmo.embed_sentence(list(tokenizer([k]))[0])) for k, _ in tqdm(us.items())}
        if emoji:
            emoji_vec, emoji2index, _ = load_emoji_vec()
            us_elmo = {k: cat_emoji_vec(k, v, emoji_vec, emoji2index) for k, v in tqdm(us.items())}
        save_pkl(us_elmo, save_path)
        return us_elmo

def prepare_data_for_feature():
    vocab = generate_vocab()
    train, val, dev_no_lab = read_data()
    return train, val, dev_no_lab, vocab

def prepare_data_loaders(num_split, batch_size=32, hier=False, elmo=False, elmo_pre=None, use_elmo_pre=False, deepmoji=False, dev_with_label=False, include_test=False):
    """ Returns a set of data loaders with shuffled train and dev set (allows overlapping samples) """
    train_data_loaders = []
    val_data_loaders = []
    test_data_loaders = []

    vocab = generate_vocab(deepmoji)
    for i in range(num_split):
        train, val, test, _ = prepare_data(batch_size=batch_size, hier=hier, elmo=elmo, elmo_pre=elmo_pre, use_elmo_pre=use_elmo_pre, deepmoji=deepmoji, is_shuffle=True, random_state=i, vocab=vocab, dev_with_label=dev_with_label, include_test=include_test)
        train_data_loaders.append(train)
        val_data_loaders.append(val)
        test_data_loaders.append(test)

    return train_data_loaders, val_data_loaders, test_data_loaders, vocab

def prepare_data_loaders_without_shuffle(batch_size=32, hier=False, elmo=False, elmo_pre=None, use_elmo_pre=False, deepmoji=False, dev_with_label=False, include_test=False):
    """ Returns a data loader for train, val, test """
    train_data_loaders = []
    val_data_loaders = []
    test_data_loaders = []

    vocab = generate_vocab(deepmoji)
    train, val, test, _ = prepare_data(batch_size=batch_size, hier=hier, elmo=elmo, elmo_pre=elmo_pre, use_elmo_pre=use_elmo_pre, deepmoji=deepmoji, is_shuffle=False, vocab=vocab, dev_with_label=dev_with_label, include_test=include_test)

    return train, val, test, vocab

def prepare_data(batch_size=32, hier=False, elmo=False, elmo_pre=None, use_elmo_pre=False, deepmoji=False, is_shuffle=False, random_state=0, vocab=None, dev_with_label=False, include_test=False):
    if vocab == None:
        vocab = generate_vocab()

    if shuffle:
        train, val, dev_no_lab = read_data(is_shuffle=is_shuffle, random_state=random_state, dev_with_label=dev_with_label, include_test=include_test)
    else:
        train, val, dev_no_lab = read_data(include_test=include_test)

    dataset_train = Dataset(train, vocab, hier=hier, elmo=elmo, elmo_pre=elmo_pre, deepmoji=deepmoji)
    data_loader_tr = torch.utils.data.DataLoader(dataset=dataset_train,
                                                    batch_size=batch_size, 
                                                    collate_fn=collate(hier, elmo, use_elmo_pre=use_elmo_pre),
                                                    shuffle=shuffle)
    
    dataset_val = Dataset(val, vocab, hier=hier, elmo=elmo, elmo_pre=elmo_pre, deepmoji=deepmoji)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                                batch_size=batch_size,
                                                shuffle=False, 
                                                collate_fn=collate(hier, elmo, use_elmo_pre=use_elmo_pre))

    dataset_dev_no_lab = Dataset(dev_no_lab, vocab, hier=hier, elmo=elmo, elmo_pre=elmo_pre, deepmoji=deepmoji)
    data_loader_dev_no_lab = torch.utils.data.DataLoader(dataset=dataset_dev_no_lab,
                                                batch_size=batch_size,
                                                shuffle=False, 
                                                collate_fn=collate(hier, elmo, use_elmo_pre=use_elmo_pre))
    
    return data_loader_tr, data_loader_val, data_loader_dev_no_lab, vocab
