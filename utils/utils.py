from utils import constant

import torch
import torch.nn as nn
import torch.utils.data

import numpy as np

import os
import pathlib

# western_happy_emojis = ["☺","🙂","😊","😀","😁","😃","😄","😆","😍","😂","😗","😙","😚","😘","😍","😉","😜","😘","😛","😝","😜","🤑","😇","👼","❤️"]
# eastern_happy_emojis = ["😀","😆","😃","😄","😺","😸","😹","😻","😽","😃","😄","☺️","😁","😀","😍","😍","😀","😂","😙","😚","😀","😁","😆","😃","😄","😁","😘","😚","😙","😗"]
# western_sad_emojis = ["☹️","🙁","😞","😟","😣","😖","😢","😭","😟","💔"]
# eastern_sad_emojis = ["😣","😖","😓","😥","😭","😢","😫","😩","😔","😔","😟"]
# western_angry_emojis = ["😠","😡"]
# eastern_angry_emojis = ["😠","😡"]

happy_emojis = ['😁', '😂', '😀', '👌', '👼', '😊', '😋', '😄', '😇', '😃', '😆', '🌹', '💋', '😗', '😻', '👉', '👫', '👭', '👬', '💑', '✌', '🐶', '🕺', '🙈', '🎁', '🙃', '✔', '💍', '😺', '🐻', '🐼', '💃', '🔥', '🐱', '🍒', '✨', '✴', '😙', '🎃', '🤘']
sad_emojis = ['😭', '😢', '💔', '😪', '😞', '😦', '🙁', '😔', '😫', '😩', '😖', '😥', '😓', '😟', '🐷', '😧', '😯', '☹', '😿', '🤒', '💖', '💰', '🚺']
angry_emojis = ['😈', '👿', '😡', '😠', '🏡', '🐺', '🐍', '🐞', '🐝', '🐬', '🐛', '😤', '☝', '🖕', '‼', '🌟', '😨', '👪', '💩', '🚬']


# happy_emojis = western_happy_emojis + eastern_happy_emojis
# sad_emojis = western_sad_emojis + eastern_sad_emojis
# angry_emojis = western_angry_emojis + eastern_angry_emojis

happy_emojis = dict((k,True) for k in happy_emojis)
sad_emojis = dict((k,True) for k in sad_emojis)
angry_emojis = dict((k,True) for k in angry_emojis)

def find_emojis(seq):
    happy_count, sad_count, angry_count = 0, 0, 0
    for char in seq:
        if char in happy_emojis:
            happy_count += 1
        if char in sad_emojis:
            sad_count += 1
        if char in angry_emojis:
            angry_count += 1
    if happy_count > 0 and sad_count == 0 and angry_count == 0:
        return "happy"
    elif sad_count > 0 and happy_count == 0 and angry_count == 0:
        return "sad"
    elif angry_count > 0 and sad_count == 0 and happy_count == 0:
        return "angry"
    else:
        return None

def evaluate(model, criterion, loader):
    model.eval()
    pred = []
    gold = []
    for X, x_len, y, ind, X_text in loader:
        # pred_prob = model(X, x_len)
        if x_len is None: pred_prob = model(X)
        else: pred_prob = model(X, x_len)
        pred.append(pred_prob[0].detach().cpu().numpy())
        gold.append(y.cpu().numpy())

    pred = np.concatenate(pred)
    gold = np.concatenate(gold)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred,gold,verbose=True)
    return microF1

def predict(model, criterion, loader, model_save_path, split=0):
    label2emotion = ["others","happy", "sad","angry"]
    model.eval()
    if split == "predict":
        file = model_save_path+"/test_{}.txt".format(split)
    else:
        file = constant.save_path+"/test_{}.txt".format(split)
    # file = constant.save_path+"test_{}.txt".format(split)
    with open(file, 'w') as the_file:
        the_file.write("id\tturn1\tturn2\tturn3\tlabel\n")
        preds_dict = {}
        indices = []
        count = 0
        for X, x_len, _, ind, X_text in loader:
            # pred_prob = model(X, x_len)
            if x_len is None: pred_prob = model(X)
            else: pred_prob = model(X, x_len)
            preds = pred_prob[1].data.max(1)[1] # max func return (max, argmax)
            for idx, text, pred in zip(ind,X_text,preds):
                preds_dict[idx] = "{}\t{}\t{}\t{}\t{}\n".format(idx,text[0],text[1],text[2],label2emotion[pred.item()])
                indices.append(idx)
            
        sorted_indices = np.argsort(-np.array(indices))[::-1]
        for idx in range(len(sorted_indices)):
            the_file.write(preds_dict[idx])
    print("FILE {} SAVED".format(file))

def predict_hier(model, criterion, loader, model_save_path, split=0, emoji_filter=False):
    label2emotion = ["others","happy", "sad","angry"]
    model.eval()
    if split == "predict":
        # file = model_save_path+"/test_{}.txt".format(split)
        # confidence_file = model_save_path+"/confidence_{}.txt".format(split)
        file = constant.save_prediction_path
        confidence_file = constant.save_confidence_path
    
        try:
            directory = os.path.dirname(constant.save_prediction_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            directory = os.path.dirname(constant.save_confidence_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
        except:
            print("mkdir error")
    else:
        file = constant.save_path+"/test_{}.txt".format(split)
        confidence_file = constant.save_path+"/test_confidence_{}.txt".format(split)
    
        directory = os.path.dirname(constant.save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    with open(confidence_file, 'w') as the_confidence_file:
        with open(file, 'w') as the_file:
            the_file.write("id\tturn1\tturn2\tturn3\tlabel\n")
            the_confidence_file.write("id\tturn1\tturn2\tturn3\tlabel\tothers\thappy\tsad\tangry\n")

            preds_dict = {}
            preds_confidence_dict = {}
            indices = []
            count = 0
            for X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text in loader:
                if x1_len is None:
                    pred_prob = model(X_1, X_2, X_3)
                else:
                    pred_prob = model(X_1, X_2, X_3, x1_len, x2_len, x3_len)

                preds = pred_prob[1].data.max(1)[1] # max func return (max, argmax)
                confidence_preds = pred_prob[1].data.max(1)[0]

                # order_idx=0
                for idx, text, pred, confidences in zip(ind,X_text,preds, pred_prob[1]):
                    new_idx = idx
                    if idx in preds_dict:
                        new_idx = 100000 + idx

                    preds_dict[new_idx] = "{}\t{}\t{}\t{}\t{}\n".format(new_idx,text[0],text[1],text[2],label2emotion[pred.item()])
                    preds_confidence_dict[new_idx] = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(new_idx,text[0],text[1],text[2],label2emotion[pred.item()],confidences[0], confidences[1], confidences[2], confidences[3])
                    # print(preds_confidence_dict[idx])
                    if emoji_filter:
                        # print(len(text), text) 
                        turn3 = text[2]
                        emotion = find_emojis(turn3)
                        if emotion is not None:
                            if emotion != label2emotion[pred.item()]:
                                print(">", turn3, emotion)
                                preds_dict[new_idx] = "{}\t{}\t{}\t{}\t{}\n".format(new_idx,text[0],text[1],text[2],emotion)

                    indices.append(new_idx)
                    # order_idx += 1

            # print(len(indices))
            sorted_indices = np.argsort(-np.array(indices))[::-1]
            print(len(preds_dict), len(preds_confidence_dict), len(sorted_indices))
            for idx in range(len(sorted_indices)):
                the_file.write(preds_dict[indices[sorted_indices[idx]]])
                the_confidence_file.write(preds_confidence_dict[indices[sorted_indices[idx]]])

    print("FILE {} SAVED".format(file))

def load_settings(arg):
    constant.model = arg.model

    # Hyperparameters
    constant.hidden_dim = arg.hidden_dim
    constant.emb_dim = arg.emb_dim
    constant.batch_size = arg.batch_size
    constant.lr = arg.lr
    constant.seed = arg.seed
    constant.num_split = arg.num_split
    constant.max_epochs = arg.max_epochs
    constant.attn = arg.attn

    USE_CUDA = arg.cuda
    UNK_idx = 0
    PAD_idx = 1
    EOS_idx = 2
    SOS_idx = 3
    CLS_idx = 4

    ### pretrained embeddings
    # emb_file = "vectors/glove/glove.6B.{}d.txt".format(str(emb_dim))
    constant.emb_map = {"glove840B~300" : "vectors/glove/glove.840B.300d.txt", "emoji~300": "vectors/emoji/emoji_embeddings_300d.txt"}

    constant.emb_file = arg.pretrain_list
    constant.emb_file_list = []
    if constant.emb_file != "":
        for emb in constant.emb_file.split(","):
            constant.emb_file_list.append(constant.emb_map[emb] + "~" + emb.split("~")[1])

    constant.emb_dim_list = []
    constant.pretrained = arg.pretrain_emb
    if(constant.pretrained):
        constant.emb_dim = arg.emb_dim
    # constant.fix_pretrain = arg.fix_pretrain

    # Double Supervision
    constant.super_ratio = arg.super_ratio
    constant.double_supervision = arg.double_supervision

    constant.save_path = arg.save_path
    constant.test = arg.test
    constant.elmo = arg.elmo
    constant.dev_with_label = arg.dev_with_label
    constant.include_test = arg.include_test

    ### LSTM
    constant.n_layers = arg.n_layers
    constant.bidirec = arg.bidirec
    constant.drop = arg.drop
    constant.attn = arg.attn
    constant.multiattn = arg.multiattn
    # constant.context = arg.context
    # constant.avgpool = arg.avgpool

    # constant.pool_kernel = arg.pool_kernel
    # constant.pool_stride = arg.pool_stride

    ### Transformer 
    constant.hop = arg.hop
    constant.heads = arg.heads
    constant.depth = arg.depth
    constant.depth_val = arg.depth_val

    constant.filter = arg.filter
    constant.label_smoothing = arg.label_smoothing
    constant.weight_sharing = arg.weight_sharing
    constant.noam = arg.noam
    constant.universal = arg.universal
    constant.act = arg.act
    constant.act_loss_weight = arg.act_loss_weight
    constant.mask = arg.mask
    constant.patient = arg.patient

    ## Meta-learn
    constant.meta_lr = 1.0

    ## eval
    constant.pred_file_path = arg.pred_file_path
    constant.ground_file_path = arg.ground_file_path

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def getMetrics(predictions, ground, verbose=False):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    one_hot = np.zeros((ground.shape[0], 4))
    one_hot[np.arange(ground.shape[0]), ground] = 1
    ground = one_hot
    label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1),num_classes=4)
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    if(verbose):
        print("True Positives per class : ", truePositives)
        print("False Positives per class : ", falsePositives)
        print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    NUM_CLASSES = 4
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        if(verbose):
            print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    if(verbose):
        print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    if(verbose):
        print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    if(verbose):
        print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.y[idx]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples