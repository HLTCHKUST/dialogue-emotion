import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn

from models.common_layer import NoamOpt
from models import HELMoEncoder, HLstmModel, HUTransformer, HDeepMoji
from utils import constant
from utils.data_reader import prepare_data, prepare_data_loaders, prepare_data_loaders_without_shuffle
from utils.utils import getMetrics, predict_hier, load_settings

from sklearn import svm
from xgboost import XGBClassifier

def load_model():
    constant.evaluate = True
    model_load_path = constant.load_model_path
    model_save_path = constant.save_path

    state = torch.load(model_load_path, map_location= lambda storage, location: storage)    
    arg = state['config']

    load_settings(arg)

    data_loader_tr, data_loader_val, data_loader_test, vocab = prepare_data_loaders_without_shuffle(
        batch_size=constant.batch_size,
        hier=True,
        elmo=constant.elmo,
        deepmoji=(constant.model=="DEEPMOJI"),
        dev_with_label=False,
        include_test=True
    )

    if constant.model == "LSTM":
        model = HLstmModel(
            vocab=vocab,
            embedding_size=constant.emb_dim,
            hidden_size=constant.hidden_dim,
            num_layers=constant.n_layers,
            is_bidirectional=constant.bidirec,
            input_dropout=constant.drop,
            layer_dropout=constant.drop,
            attentive=constant.attn,
            multiattentive=constant.multiattn,
            num_heads=constant.heads,
            total_key_depth=constant.depth,
            total_value_depth=constant.depth,
            super_ratio=constant.super_ratio,
            double_supervision=constant.double_supervision,
            context=constant.context
        )
    elif constant.model == "UTRS":
        model = HUTransformer(
            vocab=vocab,
            embedding_size=constant.emb_dim,
            hidden_size=constant.hidden_dim,
            num_layers=constant.hop,
            num_heads=constant.heads,
            total_key_depth=constant.depth,
            total_value_depth=constant.depth,
            filter_size=constant.filter,
            act=constant.act,
            input_dropout=constant.input_dropout, 
            layer_dropout=constant.layer_dropout, 
            attention_dropout=constant.attention_dropout, 
            relu_dropout=constant.relu_dropout,
        )
    elif constant.model == "ELMO":
        model = HELMoEncoder(
            H=constant.hidden_dim, L=constant.n_layers, B=constant.bidirec, C=4
        )
    elif constant.model == "DEEPMOJI":
        model = HDeepMoji(
            vocab=vocab,
            # embedding_size=constant.emb_dim,
            hidden_size=constant.hidden_dim,
            num_layers=constant.n_layers,
            max_length=700,
            input_dropout=0.0,
            layer_dropout=0.0,
            is_bidirectional=False,
            attentive=False,
        )
    else:
        print("Model is not defined")
        exit(0)

    model.load_state_dict(state['model'])
    return model, data_loader_tr, data_loader_val, data_loader_test, vocab, model_save_path

def evaluate(model, data_loader_test, vocab, model_save_path, emoji_filter=False):
    """ 
    Inputs:
        model: trained model
        data_loader_test: test data loader
        vocab: vocabulary list
    """
    if(constant.USE_CUDA): model.cuda()           
    criterion = nn.CrossEntropyLoss()
    if not emoji_filter:
        predict_hier(model, criterion, data_loader_test, model_save_path, "predict", emoji_filter=False)
    else:
        predict_hier(model, criterion, data_loader_test, model_save_path, "predict_emoji_filter", emoji_filter=True)

def evaluate_with_svm(model, data_loader_train, data_loader_valid, data_loader_test, vocab, model_save_path):
    train_x_list = []
    train_y_list = []
    test_x_list = []

    pbar = tqdm(enumerate(data_loader_train), total=len(data_loader_train))
    for i, (X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text) in pbar:
        features = model(X_1, X_2, X_3, x1_len, x2_len, x3_len, extract_feature=True) # (batch_size, feature_size)
        train_x_list.append(features.cpu().detach().numpy())
        train_y_list.append(y)
        pbar.set_description("TRAIN")

    pbar = tqdm(enumerate(data_loader_valid), total=len(data_loader_valid))
    for i, (X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text) in pbar:
        features = model(X_1, X_2, X_3, x1_len, x2_len, x3_len, extract_feature=True) # (batch_size, feature_size)
        train_x_list.append(features.cpu().detach().numpy())
        train_y_list.append(y)
        pbar.set_description("VALID")

    preds_dict = {}
    indices = []
    for X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text in data_loader_test:
        features = model(X_1, X_2, X_3, x1_len, x2_len, x3_len, extract_feature=True) # (batch_size, feature_size)
        test_x_list.append(features.cpu().detach().numpy())
        
        for idx, text in zip(ind,X_text):
            preds_dict[idx] = "{}\t{}\t{}\t{}".format(idx,text[0], text[1], text[2])
            indices.append(idx)

    train_x_list = np.concatenate(train_x_list, axis=0)
    train_y_list = np.concatenate(train_y_list, axis=0)
    test_x_list = np.concatenate(test_x_list, axis=0)
    print("train_x:", train_x_list.shape)
    print("train_y:", train_y_list.shape)
    print("test_x:", test_x_list.shape)

    print("Training SVM")
    clf = svm.SVC(gamma='scale')
    clf.fit(train_x_list, train_y_list)
    print("Predict")
    y_pred = clf.predict(test_x_list)

    assert len(y_pred) == len(indices)
    
    label2emotion = ["others","happy", "sad","angry"]
    for i in range(len(indices)):
        idx = indices[i]
        preds_dict[idx] += "\t" + label2emotion[int(y_pred[i])] + "\n"
    
    # file_path = model_save_path+"/test_svm_predict.txt"
        
    print("Print prediction to:", model_save_path)
    with open(model_save_path, 'w') as the_file:
        the_file.write("id\tturn1\tturn2\tturn3\tlabel\n")

        sorted_indices = np.argsort(-np.array(indices))[::-1]
        for idx in range(len(sorted_indices)):
            the_file.write(preds_dict[idx])

def evaluate_with_xgb(model, data_loader_train, data_loader_valid, data_loader_test, vocab, model_save_path):
    train_x_list = []
    train_y_list = []
    test_x_list = []

    pbar = tqdm(enumerate(data_loader_train), total=len(data_loader_train))
    for i, (X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text) in pbar:
        features = model(X_1, X_2, X_3, x1_len, x2_len, x3_len, extract_feature=True) # (batch_size, feature_size)
        train_x_list.append(features.cpu().detach().numpy())
        train_y_list.append(y)
        pbar.set_description("TRAIN")

    pbar = tqdm(enumerate(data_loader_valid), total=len(data_loader_valid))
    for i, (X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text) in pbar:
        features = model(X_1, X_2, X_3, x1_len, x2_len, x3_len, extract_feature=True) # (batch_size, feature_size)
        train_x_list.append(features.cpu().detach().numpy())
        train_y_list.append(y)
        pbar.set_description("VALID")

    preds_dict = {}
    indices = []
    for X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text in data_loader_test:
        features = model(X_1, X_2, X_3, x1_len, x2_len, x3_len, extract_feature=True) # (batch_size, feature_size)
        test_x_list.append(features.cpu().detach().numpy())
        
        for idx, text in zip(ind,X_text):
            preds_dict[idx] = "{}\t{}\t{}\t{}".format(idx,text[0], text[1], text[2])
            indices.append(idx)

    train_x_list = np.concatenate(train_x_list, axis=0)
    train_y_list = np.concatenate(train_y_list, axis=0)
    test_x_list = np.concatenate(test_x_list, axis=0)
    print("train_x:", train_x_list.shape)
    print("train_y:", train_y_list.shape)
    print("test_x:", test_x_list.shape)

    print("Training XGBoost")
    max_depth = constant.max_depth
    n_estimators = constant.n_estimators
    min_child_weight = constant.min_child_weight
    print('PARAMS: ',max_depth, n_estimators, min_child_weight)
    clf = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, min_child_weight=min_child_weight, n_jobs=4, tree_method="gpu_hist")
    clf.fit(train_x_list, train_y_list)
    print("Predict")
    y_pred = clf.predict(test_x_list)

    assert len(y_pred) == len(indices)
    
    label2emotion = ["others","happy", "sad","angry"]
    for i in range(len(indices)):
        idx = indices[i]
        preds_dict[idx] += "\t" + label2emotion[int(y_pred[i])] + "\n"
        
    print("Print prediction to:", model_save_path)
    with open(model_save_path, 'w') as the_file:
        the_file.write("id\tturn1\tturn2\tturn3\tlabel\n")

        sorted_indices = np.argsort(-np.array(indices))[::-1]
        for idx in range(len(sorted_indices)):
            the_file.write(preds_dict[idx])
    
if __name__ == "__main__":
    model, data_loaders_train, data_loaders_valid, data_loaders_test, vocab, model_save_path = load_model()
    print(model)

    if constant.USE_CUDA:
        model = model.cuda()

    print("USE_SVM:", constant.use_svm)
    print("USE_XGB:", constant.use_xgb)
    if constant.use_svm:
        evaluate_with_svm(model, data_loaders_train, data_loaders_valid, data_loaders_test, vocab, model_save_path)
    elif constant.use_xgb:
        evaluate_with_xgb(model, data_loaders_train, data_loaders_valid, data_loaders_test, vocab, model_save_path)
    else:
        if constant.evaluate_type == "test":
            evaluate(model, data_loaders_test, vocab, model_save_path, emoji_filter=constant.emoji_filter)
        else:
            evaluate_type = constant.evaluate_type
            if evaluate_type == "train":
                evaluate(model, data_loaders_train, vocab, model_save_path, emoji_filter=constant.emoji_filter)
            elif evaluate_type == "valid":
                evaluate(model, data_loaders_valid, vocab, model_save_path, emoji_filter=constant.emoji_filter)
            else:
                print("wrong evaluation type")