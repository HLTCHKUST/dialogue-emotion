import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn

from models.common_layer import NoamOpt
from models import ELMoEncoder, LstmModel, UTransformer
from utils import constant
from utils.data_reader import prepare_data, prepare_data_loaders
from utils.utils import evaluate, getMetrics, predict

def save_model(model, split):
    model_save_path = os.path.join(constant.save_path, 'model_{}'.format(split) )
    args = {'model':model.state_dict(), 'config':constant.arg}
    torch.save(args, model_save_path)
    print("Model saved in:",model_save_path)

def load_model():
    model_save_path = constant.load_model_path
    state = torch.load(model_save_path, map_location= lambda storage, location: storage)    
    constant.arg = state['config']
    load_settings()

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

    model = model.load_state_dict(state['model'])
    return model

def train(model, data_loader_train, data_loader_val, data_loader_test, vocab, patient=10, split=0):
    """ 
    Training loop
    Inputs:
        model: the model to be trained
        data_loader_train: training data loader
        data_loader_val: validation data loader
        vocab: vocabulary list
    Output:
        avg_best: best f1 score on validation data
    """
    if(constant.USE_CUDA): model.cuda()           
    criterion = nn.CrossEntropyLoss()
    if(constant.noam):
        opt = NoamOpt(constant.emb_dim, 1, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        opt = torch.optim.Adam(model.parameters(),lr=constant.lr)

    avg_best = 0
    cnt = 0
    for e in range(constant.max_epochs):
        model.train()
        loss_log = []
        f1_log = 0

        pbar = tqdm(enumerate(data_loader_train),total=len(data_loader_train))
        for i, (X, x_len, y, ind, X_text) in pbar:
            if constant.noam:
                opt.optimizer.zero_grad()
            else:
                opt.zero_grad()
            if x_len is None: pred_prob = model(X)
            else: pred_prob = model(X, x_len)
            
            loss = criterion(pred_prob[0], y)

            loss.backward()
            opt.step()

            ## logging 
            loss_log.append(loss.item())
            accuracy, microPrecision, microRecall, microF1 = getMetrics(pred_prob[0].detach().cpu().numpy(),y.cpu().numpy())
            f1_log += microF1
            pbar.set_description("(Epoch {}) TRAIN MICRO:{:.4f} TRAIN LOSS:{:.4f}".format((e+1), f1_log/float(i+1), np.mean(loss_log)))

        ## LOG
        if(e % 1 == 0):
            microF1 = evaluate(model, criterion, data_loader_val)
            if(microF1 > avg_best):
                avg_best = microF1
                save_model(model, split)
                predict(model, criterion, data_loader_test, "", split=split) ## print the prediction with the highest Micro-F1
                cnt = 0
            else:
                cnt += 1
            if(cnt == patient): break
            if(avg_best == 1.0): break 

            correct = 0
            loss_nb = 0

    return avg_best

if __name__ == "__main__":
    data_loaders_train, data_loaders_val, data_loaders_test, vocab = prepare_data_loaders(num_split=constant.num_split, batch_size=constant.batch_size, hier=False, elmo=constant.elmo, dev_with_label=constant.dev_with_label, include_test=constant.include_test)
    results = []

    for i in range(constant.num_split):
        data_loader_train = data_loaders_train[i]
        data_loader_val = data_loaders_val[i]
        data_loader_test = data_loaders_test[i]

        print("###### EXPERIMENT {} ######".format(i+1))
        print("(EXPERIMENT %d) Create the model" % (i+1))
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
        print(model)
        if not os.path.exists(constant.save_path):
            os.makedirs(constant.save_path)
        avg_best = train(model, data_loader_train, data_loader_val, data_loader_test, vocab, patient=constant.patient, split=i)
        results.append(avg_best)
        print("(EXPERIMENT %d) Best F1 VAL: %3.5f" % ((i+1), avg_best))

    file_summary = constant.save_path+"summary.txt"
    with open(file_summary, 'w') as the_file:
        header = "\t".join(["SPLIT_{}".format(i) for i, _ in enumerate(results)])
        the_file.write(header+"\tAVG\n")
        ris = "\t".join(["{:.4f}".format(e) for i, e in enumerate(results)])
        the_file.write(ris+"\t{:.4f}\n".format(np.mean(results)))
