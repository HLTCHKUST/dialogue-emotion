from utils import constant
import os
import numpy as np
import urllib.request
import zipfile
import os, sys
import torch
import torch.nn as nn
from allennlp.commands.elmo import ElmoEmbedder
from pytorch_pretrained_bert import BertTokenizer, BertModel
from tqdm import tqdm
from utils.emo_features import MojiModel, EmoFeatures
# from utils.emo_features import EmoFeatures, MojiModel

emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

def gen_embeddings(vocab, emb_file):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.zeros((vocab.n_words, constant.emb_dim))
    print('Embeddings: %d x %d' % (vocab.n_words, constant.emb_dim))
    if constant.emb_file is not None:
        print('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        for line in open(emb_file).readlines():
            sp = line.split()
            if(len(sp) == constant.emb_dim + 1): 
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print("Error:",sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

def multi_gen_embeddings(vocab, emb_file_list):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.zeros((vocab.n_words, constant.emb_dim))
    print('Embeddings: %d x %d' % (vocab.n_words, constant.emb_dim))
    print('Reading multiple embedding: %d' % (len(emb_file_list)))
    
    gen_vocabs = {}

    start_dim, end_dim = 0, 0
    for i in range(len(constant.emb_file_list)):
        emb_file, emb_dim = constant.emb_file_list[i].split("~")
        emb_dim = int(emb_dim)
        print('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        end_dim += emb_dim
        for line in open(emb_file).readlines():
            sp = line.split()
            if(len(sp) == emb_dim + 1): 
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]][start_dim:end_dim] = [float(x) for x in sp[1:]]
                    gen_vocabs[sp[0]] = True
            else:
                print("Error:",sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
        start_dim += emb_dim
    return embeddings, gen_vocabs

def share_embedding(vocab, pretrain=True, fix_pretrain=False):
    embedding = nn.Embedding(vocab.n_words, constant.emb_dim, padding_idx=constant.PAD_idx)
    if pretrain:
        print("Fix pretraining:", fix_pretrain)
        pre_embedding, gen_vocabs = multi_gen_embeddings(vocab,emb_file_list = constant.emb_file_list)
        constant.gen_vocabs = gen_vocabs

        embedding.weight.data.copy_(torch.FloatTensor(pre_embedding))
        if fix_pretrain:
            embedding.weight.data.requires_grad = False
        else:
            embedding.weight.data.requires_grad = True
    return embedding

def generate_elmo_features(data, vocab, split='train'):
    """ Generate ElMo features """
    if(os.path.isfile('vectors/elmo/X_{}_ELMO.out.npy'.format(split))):
        X = np.load('vectors/elmo/X_{}_ELMO.out.npy'.format(split))
        y = np.load('vectors/elmo/y_{}_ELMO.out.npy'.format(split))
        print("Pretrained Elmo vector loaded")
    else:
        elmo = ElmoEmbedder(cuda_device=5)
        X, y = [], []
        pbar = tqdm(enumerate(zip(data[0],data[1],data[2])),total=sum(1 for _ in zip(data[0],data[1],data[2])))
        for i, (_, text, label) in pbar: # zip(data[0],data[1],data[2]):
            y.append(int(emotion2label[label]))
            elmo_1 = np.array(np.mean(np.sum(elmo.embed_sentence(vocab.tokenize(text[0])),axis=0),axis=0)) # this can be done differently ask JAY
            elmo_2 = np.array(np.mean(np.sum(elmo.embed_sentence(vocab.tokenize(text[1])),axis=0),axis=0)) # this can be done differently ask JAY
            elmo_3 = np.array(np.mean(np.sum(elmo.embed_sentence(vocab.tokenize(text[2])),axis=0),axis=0)) # this can be done differently ask JAY
            feature = np.stack((elmo_1,elmo_2,elmo_3))
            X.append(feature)
        X = np.array(X)
        y = np.array(y)
        if not os.path.exists('vectors/elmo'):
            os.makedirs('vectors/elmo')
        np.save('vectors/elmo/X_{}_ELMO.out'.format(split), X) 
        np.save('vectors/elmo/y_{}_ELMO.out'.format(split), y) 

    print("Feature X Size",X.shape)
    print("Label y Size",y.shape)
    return X,y


def generate_bert_features(data, vocab, split='train'):
    """ Generate Bert features """

    if(os.path.isfile('vectors/bert/X_{}_BERT.out.npy'.format(split))):
        X = np.load('vectors/bert/X_{}_BERT.out.npy'.format(split))
        y = np.load('vectors/bert/y_{}_BERT.out.npy'.format(split))
        print("Pretrained BERT vector loaded")
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        X, y = [], []
        pbar = tqdm(enumerate(zip(data[0],data[1],data[2])),total=sum(1 for _ in zip(data[0],data[1],data[2])))
        for i, (_, text, label) in pbar: # zip(data[0],data[1],data[2]):
            # Convert inputs to PyTorch tensors
            tt = tokenizer.tokenize(text[0])
            if(len(tt)==0): tt.append('[UNK]')
            tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tt)])
            segments_tensors = torch.tensor([[0 for _ in range(len(tt))]])
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
            bert_1 = np.sum(encoded_layers[-1].detach().numpy(),axis=1)

            tt = tokenizer.tokenize(text[1])
            if(len(tt)==0): tt.append('[UNK]')
            tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tt)])
            segments_tensors = torch.tensor([[0 for _ in range(len(tt))]])
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
            bert_2 = np.sum(encoded_layers[-1].detach().numpy(),axis=1)

            tt = tokenizer.tokenize(text[2])
            if(len(tt)==0): tt.append('[UNK]')
            tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tt)])
            segments_tensors = torch.tensor([[0 for _ in range(len(tt))]])
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
            bert_3 = np.sum(encoded_layers[-1].detach().numpy(),axis=1)    

            feature = np.stack((bert_1,bert_2,bert_3))
            X.append(feature)
            y.append(int(emotion2label[label]))
        X = np.array(X)
        y = np.array(y)
        if not os.path.exists('vectors/bert'):
            os.makedirs('vectors/bert')
        np.save('vectors/bert/X_{}_BERT.out'.format(split), X) 
        np.save('vectors/bert/y_{}_BERT.out'.format(split), y) 

    print("Feature X Size",X.shape)
    print("Label y Size",y.shape)
    return X,y


def generate_emo2vec_features(data, vocab, mode):
    """ Generate Emoji features """
    # constant.emb_dim can be 50 100 200 300
    emo_feat = EmoFeatures(vocab, vocab.tt)
    X, y = [], []
    for _, text, label in zip(data[0],data[1],data[2]): # text

        feature_1 = emo_feat.embedding(text[0],mode=mode) #+ 
        feature_2 = emo_feat.embedding(text[1],mode=mode) #+ 
        feature_3 = emo_feat.embedding(text[2],mode=mode) #+ 

        feature = np.stack( (feature_1,feature_2,feature_3))
        X.append(feature)
        y.append(emotion2label[label])
    X = np.array(X)
    y = np.array(y)
    print("Feature X Size",X.shape)
    print("Label y Size",y.shape)
    return X,y


def generate_deepmoji_features(data, vocab, mode, split):
    """ Generate Emoji features """
    if(os.path.isfile('vectors/deepmoji/X_{}_deepmoji_pred.out.npy'.format(split))):
        if(mode=="pred"):
            X = np.load('vectors/deepmoji/X_{}_deepmoji_pred.out.npy'.format(split))
            y = np.load('vectors/deepmoji/y_{}_deepmoji_pred.out.npy'.format(split))
        elif(mode=="feat"):
            X = np.load('vectors/deepmoji/X_{}_deepmoji_feat.out.npy'.format(split))
            y = np.load('vectors/deepmoji/y_{}_deepmoji_feat.out.npy'.format(split))
        print("Pretrained deepmoji vector loaded")
    else:
        deepmoji_feat = MojiModel()
        X_pred, y_pred = [], []
        X_feat, y_feat = [], []
        pbar = tqdm(enumerate(zip(data[0],data[1],data[2])),total=sum(1 for _ in zip(data[0],data[1],data[2])))
        for i, (_, text, label) in pbar: # zip(data[0],data[1],data[2]):

            feature_1_pred = deepmoji_feat.predict(text[0]) #+ 
            feature_2_pred = deepmoji_feat.predict(text[1]) #+ 
            feature_3_pred = deepmoji_feat.predict(text[2]) #+ 

            feature_1_feat = deepmoji_feat.moji_feat(text[0]) #+ 
            feature_2_feat = deepmoji_feat.moji_feat(text[1]) #+ 
            feature_3_feat = deepmoji_feat.moji_feat(text[2]) #+ 

            feature_pred = np.stack( (feature_1_pred,feature_2_pred,feature_3_pred))
            feature_feat = np.stack( (feature_1_feat,feature_2_feat,feature_3_feat))
            X_pred.append(feature_pred)
            y_pred.append(emotion2label[label])
            X_feat.append(feature_feat)
            y_feat.append(emotion2label[label])

        if not os.path.exists('vectors/deepmoji'):
            os.makedirs('vectors/deepmoji')
        np.save('vectors/deepmoji/X_{}_deepmoji_pred.out'.format(split), np.array(X_pred)) 
        np.save('vectors/deepmoji/y_{}_deepmoji_pred.out'.format(split), np.array(y_pred)) 
        np.save('vectors/deepmoji/X_{}_deepmoji_feat.out'.format(split), np.array(X_feat)) 
        np.save('vectors/deepmoji/y_{}_deepmoji_feat.out'.format(split), np.array(y_feat))         
        if(mode=="pred"):
            X = np.array(X_pred)
            y = np.array(y_pred)
        elif(mode=="feat"):
            X = np.array(X_feat)
            y = np.array(y_feat)

    print("Feature X Size",X.shape)
    print("Label y Size",y.shape)
    return X,y

def generate_emoji_features(data, vocab, mode):
    """ Generate Emoji features """
    # constant.emb_dim can be 50 100 200 300
    emb_emoji = gen_embeddings(vocab,emb_file = "vectors/emoji/emoji_embeddings_{}d.txt".format(str(constant.emb_dim)))
    X, y = [], []
    for _, text, label in zip(data[0],data[1],data[2]): # text
        if mode == "sum":
            feature_1 = np.sum([emb_emoji[vocab.word2index[t]] for t in vocab.tokenize(text[0])],axis=0).squeeze() #+ 
            feature_2 = np.sum([emb_emoji[vocab.word2index[t]] for t in vocab.tokenize(text[1])],axis=0).squeeze() #+ 
            feature_3 = np.sum([emb_emoji[vocab.word2index[t]] for t in vocab.tokenize(text[2])],axis=0).squeeze() #+ 
        elif mode == "avg":
            feature_1 = np.mean([emb_emoji[vocab.word2index[t]] for t in vocab.tokenize(text[0])],axis=0).squeeze() #+ emb_emoji[vocab.word2index[t]]
            feature_2 = np.mean([emb_emoji[vocab.word2index[t]] for t in vocab.tokenize(text[1])],axis=0).squeeze() #+ emb_emoji[vocab.word2index[t]]
            feature_3 = np.mean([emb_emoji[vocab.word2index[t]] for t in vocab.tokenize(text[2])],axis=0).squeeze() #+ emb_emoji[vocab.word2index[t]]
        else:
            print("GLOVE loader mode is wrong. Select sum or avg")      
        feature = np.stack( (feature_1,feature_2,feature_3))
        X.append(feature)
        y.append(emotion2label[label])
    X = np.array(X)
    y = np.array(y)
    print("Feature X Size",X.shape)
    print("Label y Size",y.shape)
    return X,y

def download_glove(ty="w2v"):
    if not os.path.exists('vectors/glove'):
        os.makedirs('vectors/glove')
    print('Beginning download glove:',ty)

    if(ty=="w2v"):
        url = 'http://nlp.stanford.edu/data/wordvecs/glove.6B.zip'  
        urllib.request.urlretrieve(url, 'vectors/glove/glove.6B.zip')  
        print('Finished to download glove:',ty)
        zip_ref = zipfile.ZipFile('vectors/glove/glove.6B.zip', 'r')
        zip_ref.extractall('vectors/glove/')
        zip_ref.close()

    elif(ty=="twitter"):
        url = 'http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip'  
        urllib.request.urlretrieve(url, 'vectors/glove/glove.twitter.27B.zip')  
        print('Finished to download glove:',ty)
        zip_ref = zipfile.ZipFile('vectors/glove/glove.twitter.27B.zip', 'r')
        zip_ref.extractall('vectors/glove/')
        zip_ref.close()

    elif(ty=="common"):
        url = 'http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip'  
        urllib.request.urlretrieve(url, 'vectors/glove/glove.840B.300d.zip')  
        print('Finished to download glove:',ty)
        zip_ref = zipfile.ZipFile('vectors/glove/glove.840B.300d.zip', 'r')
        zip_ref.extractall('vectors/glove/')
        zip_ref.close()

def generate_glove_features(data, vocab, mode, ty):
    """ Generate Glove features """
    if(ty=="w2v"):
        if(not os.path.isfile("vectors/glove/glove.6B.{}d.txt".format(str(constant.emb_dim)))): download_glove(ty)
        embeddings = gen_embeddings(vocab, emb_file = "vectors/glove/glove.6B.{}d.txt".format(str(constant.emb_dim))) 
    elif(ty=="twitter"):
        if(not os.path.isfile("vectors/glove/glove.twitter.27B.{}d.txt".format(str(constant.emb_dim)))): download_glove(ty)
        embeddings = gen_embeddings(vocab, emb_file = "vectors/glove/glove.twitter.27B.{}d.txt".format(str(constant.emb_dim))) 
    elif(ty=="common"):
        if(not os.path.isfile("vectors/glove/glove.840B.300d.txt") ): download_glove(ty)
        embeddings = gen_embeddings(vocab, emb_file = "vectors/glove/glove.840B.300d.txt") 
    else:
        print("Select the correct ty between: [w2v,twitter,common]")

    X, y = [], []
    for _, text, label in zip(data[0],data[1],data[2]): # text
        if mode == "sum":
            feature_1 = np.sum([embeddings[vocab.word2index[t]] for t in vocab.tokenize(text[0])],axis=0).squeeze() #+ emb_emoji[vocab.word2index[t]]
            feature_2 = np.sum([embeddings[vocab.word2index[t]] for t in vocab.tokenize(text[1])],axis=0).squeeze() #+ emb_emoji[vocab.word2index[t]]
            feature_3 = np.sum([embeddings[vocab.word2index[t]] for t in vocab.tokenize(text[2])],axis=0).squeeze() #+ emb_emoji[vocab.word2index[t]]
        elif mode == "avg":
            feature_1 = np.mean([embeddings[vocab.word2index[t]] for t in vocab.tokenize(text[0])],axis=0).squeeze() #+ emb_emoji[vocab.word2index[t]]
            feature_2 = np.mean([embeddings[vocab.word2index[t]] for t in vocab.tokenize(text[1])],axis=0).squeeze() #+ emb_emoji[vocab.word2index[t]]
            feature_3 = np.mean([embeddings[vocab.word2index[t]] for t in vocab.tokenize(text[2])],axis=0).squeeze() #+ emb_emoji[vocab.word2index[t]]
        else:
            print("GLOVE loader mode is wrong. Select sum or avg")
        feature = np.stack( (feature_1,feature_2,feature_3))
        X.append(feature)
        y.append(emotion2label[label])
    X = np.array(X)
    y = np.array(y)
    print("Feature X Size",X.shape)
    print("Label y Size",y.shape)
    return X,y


def get_feature(data, vocab, feature_list=['glove'], mode=['sum'], ty=['w2v'], split="train"):
    ## TODO add emo2vec and deepmoji
    ## predefine X and y
    X, y = [], []

    for i, f in enumerate(feature_list):
        if(f=='glove'):
            Xi, yi = generate_glove_features(data,vocab,mode[i],ty[i])
        elif(f=='emoji'):
            Xi, yi = generate_emoji_features(data,vocab,mode[i])
        elif(f=='emo2vec'):
            Xi, yi = generate_emo2vec_features(data,vocab,mode[i]) 
        elif(f=='elmo'):
            Xi, yi = generate_elmo_features(data,vocab,split)   
        elif(f=='bert'):
            Xi, yi = generate_bert_features(data,vocab,split)   
        elif(f=='deepmoji'):
            if(mode[i]=='sum'): mode[i] = 'pred'
            if(mode[i]=='avg'): mode[i] = 'feat'
            Xi, yi = generate_deepmoji_features(data,vocab,mode[i],split) 
        
        ## concatenate the features
        ## For bert has 4 dimensions, the feature should be reshaped first.
        if f == "bert":
            Xi = np.squeeze(Xi,axis = 2)
            pass

        if X==[]:
            X = Xi
            y = yi
        else:
            X = np.concatenate((X, Xi), axis = 2)
            ## Because for every yi, they exactly the same, we only do assignment once.
    return X, y
