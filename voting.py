from utils.data_reader import prepare_data, prepare_data_loaders
from utils import constant
from utils.utils import getMetrics

from models.lstm_model import HLstmModel
from models.transformer import HUTransformer

import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import os

import pandas as pd
import numpy as np
import os 
import math

import os.path

voting_dir_list = constant.voting_dir_list
voting_threshold = constant.voting_threshold
save_path = constant.save_path

label = {"others":0, "happy":1, "sad":2, "angry":3}
label2emotion = ["others", "happy", "sad", "angry"]

def voting_with_threshold():
    print("voting with threshold")
    classifier_pred = []
    df_list = []
    for i in range(len(voting_dir_list)):
        directory = voting_dir_list[i]
        print("Directory:", directory)
        pred = []
        summary_path = directory + "/summary.txt"
        print(summary_path)
        if os.path.exists(summary_path):
            with open(summary_path, "r") as summary_file:
                skip = True
                for line in summary_file:
                    if skip:
                        skip = False                    
                        continue
                    pred = [float(num) for num in line.split("\t")]
                    break
        else:
            continue

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(">", directory+"/"+filename)
            if filename.startswith("test"): 
                test_id = filename.replace(".txt","").replace("test_","")
                if pred[int(test_id)] > voting_threshold:
                    print(pred[int(test_id)], "threshold >", voting_threshold)
                    df = pd.read_csv(directory+"/"+filename, delimiter='\t')
                    classifier_pred.append(df['label'].values)
                    cnt_non_other = 0
                    cnt = 0 
                    for d in df['label'].values:
                        if(label[d]!=0):        
                            cnt_non_other+=1
                        else:
                            cnt+=1
                    print("Non Other:",cnt_non_other,"Other:",cnt)
                    df_list.append(df)
                else:
                    print(pred[int(test_id)], "lower than threshold", voting_threshold)
    return classifier_pred, df_list

def voting():
    print("voting")
    classifier_pred = []
    df_list = []
    for i in range(len(voting_dir_list)):
        directory = voting_dir_list[i]
        print("Directory:", directory)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(">", directory+"/"+filename)
            if filename.startswith("test") and os.path.isfile(directory + "/" + filename): 
                df = pd.read_csv(directory+"/"+filename, delimiter='\t')
                classifier_pred.append(df['label'].values)
                cnt_non_other = 0
                cnt = 0 
                for d in df['label'].values:
                    if(label[d]!=0):        
                        cnt_non_other+=1
                    else:
                        cnt+=1
                print("Non Other:",cnt_non_other,"Other:",cnt)
                df_list.append(df)
    return classifier_pred, df_list

if voting_threshold > 0:
    classifier_pred, df_list = voting_with_threshold()
else:
    classifier_pred, df_list = voting()

classifier_pred = np.array(classifier_pred).transpose()
print("number of prediction", len(df_list))
print("prediction:", classifier_pred.shape)

cnt_non_other = 0
cnt = 0 
voting_prediction = []

with open("dist.txt", "w") as out:
    out.write("others\thappy\tsad\tangry\n")
    # print(classifier_pred)
    for r in classifier_pred:
        weight_class = [0,0,0,0] 
        # print(r)
        r_number = [label[e] for i, e in enumerate(r)]
        # print(r_number) 
        bin_ = np.bincount(np.array(r_number), None, 4)
        # print(bin_)
        out.write("{}\t{}\t{}\t{}\n".format(bin_[0], bin_[1], bin_[2], bin_[3]))
        voting_prediction.append(label2emotion[np.argmax(bin_)])
        if(np.argmax(bin_)!= 0): 
            cnt_non_other+=1
        else:
            cnt+=1
print("Non Other:",cnt_non_other,"Other:",cnt)

# df = pd.read_csv(constant.save_path+"test_0.txt", delimiter='\t')
print("Print prediction")
df = df_list[0]
df['label'] = voting_prediction
df.to_csv(save_path, index=None, sep='\t')