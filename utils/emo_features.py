import numpy as np
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.model_def import torchmoji_feature_encoding
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
import json
import torch
import emoji
import os
import torch.nn as nn

EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')

class MojiModel(nn.Module):

    def __init__(self, use_cuda=True):
        super(MojiModel, self).__init__()
        self.use_cuda = use_cuda
        self.EMOJIS = EMOJIS
        self.emoji_model = torchmoji_emojis(PRETRAINED_PATH)
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)
        self.tokenizer = SentenceTokenizer(vocabulary, 100)
        print(self.emoji_model)
        self.feat_model = torchmoji_feature_encoding(PRETRAINED_PATH)
        if use_cuda:
            self.emoji_model = self.emoji_model.cuda()
            self.feat_model = self.feat_model.cuda()

    def predict(self, input_txt):
        input_txt = [input_txt]
        tokenized, _, _ = self.tokenizer.tokenize_sentences(input_txt)
        if self.use_cuda:
            tokenized = torch.cuda.LongTensor(tokenized.astype('int32'))
        prob = self.emoji_model(tokenized)[0]
        return prob

    def moji_feat(self, input_txt):
        input_txt = [input_txt]
        tokenized, _, _ = self.tokenizer.tokenize_sentences(input_txt)
        if self.use_cuda:
            tokenized = torch.cuda.LongTensor(tokenized.astype('int32'))
        return self.feat_model(tokenized)[0]

    def to_emoji(self, idx):
        return emoji.emojize(self.EMOJIS[idx], use_aliases=True)

    # def top_i(self, input_txt, i=1):
    #     prob = self.predict(input_txt)
    #     top_i_idxs = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)[:i]
    #     return [(self.to_emoji(idx), prob[idx]) for idx in top_i_idxs]
                                                                        

def get_emo_embedding(lang):

    if not os.path.exists("vectors/emo2vec/vocab.pkl") or not os.path.exists("vectors/emo2vec/emo2vec.pkl"):
        from utils.download_google_drive import download_file_from_google_drive
        file_id = '1K0RPGSlBHOng4NN4Jkju_OkYtrmqimLi'
        destination = 'vectors/emo2vec.zip'
        download_file_from_google_drive(file_id, destination)
        print("start unzipping")
        import zipfile
        zip_ref = zipfile.ZipFile(destination, 'r')
        zip_ref.extractall("vectors/")
        zip_ref.close()
        os.remove(destination)
        
    import pickle
    emo_size = 100
    emo2vec_vocab = "vectors/emo2vec/vocab.pkl"
    emo2vec_emb = "vectors/emo2vec/emo2vec.pkl"
    with open(emo2vec_vocab, 'rb') as f:
        emo_word2id = pickle.load(f, encoding="latin1")["word2id"]

    with open(emo2vec_emb,'rb') as f:
        emo_embedding = pickle.load(f, encoding='latin1')

    # new_embedding = np.zeros((100, emb_size))
    new_embedding = np.zeros((lang.n_words, emo_size))
    for i in range(lang.n_words):
    # for i in range(100):
        if emo_size > 0 and lang.index2word[i] in emo_word2id:
            new_embedding[i] = emo_embedding[emo_word2id[lang.index2word[i]]]

    return new_embedding


class EmoFeatures(object):
    def __init__(self, vocab, tokenizer):
        self.tokenizer = tokenizer
        self.emo_emb = get_emo_embedding(vocab)
        self.word2index = vocab.word2index
        self.index2word = vocab.index2word

    def embedding(self, sent, mode="full"):
        emb = []
        for w in self.tokenizer.tokenize(sent):
            if w in self.word2index:
                emb.append(self.emo_emb[self.word2index[w]])
            else:
                emb.append(np.zeros((100,)))
        if mode == "sum":
            return np.sum(emb, axis=0)
        elif mode == "avg" or "average":
            return np.mean(emb, axis=0)
        elif mode == "max":
            return np.max(np.array(emb), axis=0)
        else:
            raise ValueError("invalid mode arguments")