import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn

from models.common_layer import NoamOpt
from models import ELMoEncoder, LstmModel, UTransformer
from utils import constant
from utils.data_reader import prepare_data, prepare_data_loaders
from utils.utils import getMetrics, predict, load_settings


def load_model():
    model_load_path = constant.load_model_path
    model_save_path = constant.save_path
    state = torch.load(model_load_path, map_location=lambda storage, location: storage)
    arg = state['config']
    load_settings(arg)

    data_loaders_train, data_loaders_val, data_loaders_test, vocab = prepare_data_loaders(
        num_split=1, batch_size=constant.batch_size, hier=False, elmo=constant.elmo, dev_with_label=False, include_test=True)

    if constant.model == "LSTM":
        model = LstmModel(vocab=vocab,
                          embedding_size=constant.emb_dim,
                          hidden_size=constant.hidden_dim,
                          num_layers=constant.n_layers,
                          is_bidirectional=constant.bidirec,
                          input_dropout=constant.drop,
                          layer_dropout=constant.drop,
                          attentive=constant.attn)
    elif constant.model == "UTRS":
        model = UTransformer(vocab=vocab,
                             embedding_size=constant.emb_dim,
                             hidden_size=constant.hidden_dim,
                             num_layers=constant.hop,
                             num_heads=constant.heads,
                             total_key_depth=constant.depth,
                             total_value_depth=constant.depth,
                             filter_size=constant.filter,
                             act=constant.act)
    elif constant.model == "ELMO":
        model = ELMoEncoder(C=4)
    else:
        print("Model is not defined")
        exit(0)

    model.load_state_dict(state['model'])
    return model, data_loaders_test, vocab, model_save_path


def evaluate(model, data_loader_test, vocab, model_save_path):
    """ 
    Inputs:
        model: trained model
        data_loader_test: test data loader
        vocab: vocabulary list
    """
    if(constant.USE_CUDA):
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    # print the prediction with the highest Micro-F1
    predict(model, criterion, data_loader_test, model_save_path, "predict")


if __name__ == "__main__":
    model, data_loaders_test, vocab, model_save_path = load_model()
    print(model)
    evaluate(model, data_loaders_test[0], vocab, model_save_path)
