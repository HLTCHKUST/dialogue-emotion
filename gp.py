import matplotlib
matplotlib.use('Agg')
import os
import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
from models import HUTransformer
from utils import constant
from utils.data_reader import prepare_data
import numpy as np
import torch
import torch.nn as nn
from main_hier import train
import subprocess
import sys
from io import StringIO
import pandas as pd


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


# Optimization objective 
def h_trs(parameters):
    parameters = parameters[0]
    constant.lr = parameters[0]
    constant.input_dropout = parameters[1]
    constant.layer_dropout = parameters[2]
    constant.attention_dropout = parameters[3]
    constant.relu_dropout = parameters[4]
    constant.emb_dim = int(parameters[5])
    constant.hidden_dim = int(constant.emb_dim)
    constant.hop =  int(parameters[6])
    constant.heads = int(parameters[7])
    constant.depth_key = int(parameters[8])
    constant.depth_val = int(parameters[9])
    constant.filter = int(parameters[10])
    constant.batch_size = int(parameters[11])
    constant.device = 3
    data_loaders_tr, data_loaders_val, data_loaders_test, vocab = prepare_data(batch_size=constant.batch_size, hier=True, dev_with_label=True)
    model = HUTransformer(vocab=vocab, 
            embedding_size=constant.emb_dim, 
            hidden_size=constant.hidden_dim, 
            num_layers=constant.hop,
            num_heads=constant.heads, 
            total_key_depth=constant.depth_key, 
            total_value_depth=constant.depth_val,
            filter_size=constant.filter,
            input_dropout=constant.input_dropout, 
            layer_dropout=constant.layer_dropout, 
            attention_dropout=constant.attention_dropout, 
            relu_dropout=constant.relu_dropout, 
            use_mask=False)
    constant.save_path = "save/{}/".format(gp_folder) + str(parameters) +"/"
    if not os.path.exists(constant.save_path):
        os.makedirs(constant.save_path)

    avg_best = train(model, data_loaders_tr, data_loaders_val, data_loaders_test, vocab, patient=10, split=0, verbose=False)
    print("PARAM", parameters[0])
    print("bzs",constant.batch_size)
    print("lr",constant.lr)
    print("embedding_size:",constant.emb_dim) 
    print("hidden_size:",constant.hidden_dim) 
    print("num_layers:",constant.hop)
    print("num_heads:",constant.heads) 
    print("total_key_depth:",constant.depth_key) 
    print("total_value_depth:",constant.depth_val)
    print("filter_size:",constant.filter)
    print("input_dropout:",constant.input_dropout) 
    print("layer_dropout:",constant.layer_dropout) 
    print("attention_dropout:",constant.attention_dropout) 
    print("relu_dropout:",constant.relu_dropout)
    print("Results F1",avg_best)
    return avg_best

if __name__ == '__main__':

    gp_folder = "gp_with_dev_last"
    bds = [ {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.00001, 0.005)},
            {'name': 'input_dropout', 'type': 'continuous', 'domain': (0.0, 0.3)},
            {'name': 'layer_dropout', 'type': 'continuous', 'domain': (0.0, 0.3)},
            {'name': 'attention_dropout', 'type': 'continuous', 'domain': (0.0, 0.3)},
            {'name': 'relu_dropout', 'type': 'continuous', 'domain': (0.0, 0.3)},
            {'name': 'emb_dim', 'type': 'discrete', 'domain': tuple((i for i in range(60, 500+1))) },
            {'name': 'hop', 'type': 'discrete', 'domain': tuple((i for i in range(1, 10+1)))},
            {'name': 'heads', 'type': 'discrete', 'domain': tuple((i for i in range(1, 10+1)))},
            {'name': 'depth_key', 'type': 'discrete', 'domain': tuple((i for i in range(20, 80+1)))},
            {'name': 'depth_val', 'type': 'discrete', 'domain': tuple((i for i in range(20, 80+1)))},
            {'name': 'filter', 'type': 'discrete', 'domain': tuple((i for i in range(60, 300+1)))},
            {'name': 'batch_size', 'type': 'discrete', 'domain': tuple((i for i in range(32, 64+1)))}]

    X_init = np.array([[0.001,0.0,0.0,0.0,0.0,100,6,4,40,40,50,32]])
    Y_init = h_trs(X_init)
    optimizer = BayesianOptimization(f=h_trs, 
                                    domain=bds,
                                    model_type='GP',
                                    acquisition_type ='EI',
                                    acquisition_jitter = 0.05,
                                    exact_feval=False, 
                                    maximize=True,
                                    X=X_init,
                                    Y=np.array([[Y_init]]),
                                    verbosity_model=True)

    # # Only 20 iterations because we have 5 initial random points
    optimizer.run_optimization(max_iter=100,verbosity=True,report_file="save/{}/report.txt".format(gp_folder))
    optimizer.save_evaluations(evaluations_file="save/{}/evaluation.txt".format(gp_folder))
    optimizer.plot_acquisition(filename="save/{}/acquisition.pdf".format(gp_folder))
    optimizer.plot_convergence(filename="save/{}/convergence.pdf".format(gp_folder))


