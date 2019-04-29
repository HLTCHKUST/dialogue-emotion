import os

import dill as pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from models.common_layer import NoamOpt
from models import HELMoEncoder, HLstmModel, HUTransformer, HDeepMoji
from utils import constant
from utils.data_reader import extract_elmo, prepare_data, prepare_data_loaders
from utils.utils import getMetrics


def evaluate(model, criterion, loader, verbose=True):
    model.eval()
    pred = []
    gold = []
    for X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text in loader:
        if x1_len is None:
            pred_prob = model(X_1, X_2, X_3)
        else:
            pred_prob = model(X_1, X_2, X_3, x1_len, x2_len, x3_len)
        # pred_prob = model(X_1,X_2,X_3,x1_len,x2_len,x3_len)
        pred.append(pred_prob[0].detach().cpu().numpy())
        gold.append(y.cpu().numpy())

    pred = np.concatenate(pred)
    gold = np.concatenate(gold)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(pred, gold, verbose)
    return microF1


def predict(model, criterion, loader, split=0):
    label2emotion = ["others", "happy", "sad", "angry"]
    model.eval()
    file = constant.save_path + "test_{}.txt".format(split)
    with open(file, "w") as the_file:
        the_file.write("id\tturn1\tturn2\tturn3\tlabel\n")
        preds_dict = {}
        indices = []
        count = 0
        for X_1, X_2, X_3, x1_len, x2_len, x3_len, _, ind, X_text in loader:
            if x1_len is None:
                pred_prob = model(X_1, X_2, X_3)
            else:
                pred_prob = model(X_1, X_2, X_3, x1_len, x2_len, x3_len)
            # pred_prob = model(X_1, X_2, X_3, x1_len, x2_len, x3_len)
            preds = pred_prob[1].data.max(1)[1]  # max func return (max, argmax)
            for idx, text, pred in zip(ind, X_text, preds):
                preds_dict[idx] = "{}\t{}\t{}\t{}\t{}\n".format(
                    idx, text[0], text[1], text[2], label2emotion[pred.item()]
                )
                indices.append(idx)

        sorted_indices = np.argsort(-np.array(indices))[::-1]
        for idx in range(len(sorted_indices)):
            the_file.write(preds_dict[idx])
    # print("FILE {} SAVED".format(file))

def save_model(model, split):
    model_save_path = os.path.join(constant.save_path, 'model_{}'.format(split) )
    args = {'model':model.state_dict(), 'config':constant.arg}
    torch.save(args, model_save_path)
    print("Model saved in:",model_save_path)

def load_model(model, split):
    model_save_path = os.path.join(constant.save_path, 'model_{}'.format(split) )
    state = torch.load(model_save_path, map_location= lambda storage, location: storage)    
    model = model.load_state_dict(state['model'])
    constant.arg = state['config']
    return model

def train(
    model,
    data_loader_train,
    data_loader_val,
    data_loader_test,
    vocab,
    patient=10,
    split=0,
    verbose=True,
):
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
    if constant.USE_CUDA:
        device = torch.device("cuda:{}".format(constant.device))
        model.to(device)
    criterion = nn.CrossEntropyLoss()
    if constant.noam:
        opt = NoamOpt(
            constant.emb_dim,
            1,
            4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=constant.lr)

    ## TRAINING LOOP
    avg_best = 0
    cnt = 0
    for e in range(constant.max_epochs):
        model.train()
        loss_log = []
        f1_log = []

        pbar = tqdm(enumerate(data_loader_train), total=len(data_loader_train))
        for i, (X_1, X_2, X_3, x1_len, x2_len, x3_len, y, ind, X_text) in pbar:
            if constant.noam:
                opt.optimizer.zero_grad()
            else:
                opt.zero_grad()
            if x1_len is None:
                pred_prob = model(X_1, X_2, X_3)
            else:
                pred_prob = model(X_1, X_2, X_3, x1_len, x2_len, x3_len)

            if constant.double_supervision:
                loss = (1-constant.super_ratio)*criterion(pred_prob[0],y) + constant.super_ratio*criterion(pred_prob[2],y)
            else:
                loss = criterion(pred_prob[0], y)

            if constant.act:
                R_t = pred_prob[2][0]
                N_t = pred_prob[2][1]
                p_t = R_t + N_t
                avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
                loss += constant.act_loss_weight * avg_p_t.item()
            loss.backward()
            opt.step()
            ## logging
            loss_log.append(loss.item())
            accuracy, microPrecision, microRecall, microF1 = getMetrics(
                pred_prob[0].detach().cpu().numpy(), y.cpu().numpy()
            )
            f1_log.append(microF1)
            pbar.set_description(
                "(Epoch {}) TRAIN MICRO:{:.4f} TRAIN LOSS:{:.4f}".format(
                    (e + 1), np.mean(f1_log), np.mean(loss_log)
                )
            )

        ## LOG
        if e % 1 == 0:
            microF1 = evaluate(model, criterion, data_loader_val, verbose)
            if microF1 > avg_best:
                avg_best = microF1
                save_model(model, split)
                predict(
                    model, criterion, data_loader_test, split
                )  ## print the prediction with the highest Micro-F1
                cnt = 0
            else:
                cnt += 1
            if cnt == patient:
                break
            if avg_best == 1.0:
                break

            correct = 0
            loss_nb = 0

    return avg_best


if __name__ == "__main__":
    elmo_pre = extract_elmo(emoji=constant.emoji) if constant.use_elmo_pre else None

    data_loaders_tr, data_loaders_val, data_loaders_test, vocab = prepare_data_loaders(
        num_split=constant.num_split,
        batch_size=constant.batch_size,
        hier=True,
        elmo=constant.elmo,
        elmo_pre=elmo_pre,
        use_elmo_pre=constant.use_elmo_pre,
        deepmoji=(constant.model=="DEEPMOJI"),
        dev_with_label=constant.dev_with_label,
        include_test=constant.include_test
    )
    results = []
    for i in range(constant.num_split):
        data_loader_tr = data_loaders_tr[i]
        data_loader_val = data_loaders_val[i]
        data_loader_test = data_loaders_test[i]

        print("###### EXPERIMENT {} ######".format(i + 1))
        print("(EXPERIMENT %d) Create the model" % (i + 1))
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
                context=constant.context,
                pool=constant.pool,
                pool_kernel=constant.pool_kernel,
                pool_stride=constant.pool_stride
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
            I = 3072 if constant.use_elmo_pre else 1024
            if constant.emoji:
                I += 300

            model = HELMoEncoder(
                I=I,
                H=constant.hidden_dim, 
                L=constant.n_layers, 
                bi=constant.bidirec, 
                C=4, 
                mlp=constant.mlp, 
                pre=constant.use_elmo_pre,
                attentive=constant.attn,
                multiattentive=constant.multiattn,
                num_heads=constant.heads,
                double_supervision=constant.double_supervision,
            )
        elif constant.model == "DEEPMOJI":
            model = HDeepMoji(
                vocab=vocab,
                # embedding_size=constant.emb_dim,
                hidden_size=constant.hidden_dim,
                num_layers=constant.n_layers,
                max_length=700,
                input_dropout=constant.drop,
                layer_dropout=constant.drop,
                is_bidirectional=constant.bidirec,
                attentive=constant.attn,
            )
        else:
            print("Model is not defined")
            exit(0)

        print(model)
        if not os.path.exists(constant.save_path):
            os.makedirs(constant.save_path)
        avg_best = train(
            model,
            data_loader_tr,
            data_loader_val,
            data_loader_test,
            vocab,
            patient=constant.patient,
            split=i,
        )
        results.append(avg_best)
        print("(EXPERIMENT %d) Best F1 VAL: %3.5f" % ((i + 1), avg_best))

    file_summary = constant.save_path + "summary.txt"
    with open(file_summary, "w") as the_file:
        header = "\t".join(["SPLIT_{}".format(i) for i, _ in enumerate(results)])
        the_file.write(header + "\tAVG\n")
        ris = "\t".join(["{:.4f}".format(e) for i, e in enumerate(results)])
        the_file.write(ris + "\t{:.4f}\n".format(np.mean(results)))

