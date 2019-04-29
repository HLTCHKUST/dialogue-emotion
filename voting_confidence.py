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
save_prediction_path = constant.save_prediction_path
save_confidence_path = constant.save_confidence_path

label = {"others":0, "happy":1, "sad":2, "angry":3}
label2emotion = ["others", "happy", "sad", "angry"]

def voting():
    print("voting")
    happy_pred = []
    sad_pred = []
    angry_pred = []
    others_pred = []

    df_list = []
    pred_num = 0
    for i in range(len(voting_dir_list)):
        directory = voting_dir_list[i]
        print("Directory:", directory)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(">", directory+"/"+filename)
            if filename.startswith("test_confidence"): 
                df = pd.read_csv(directory+"/"+filename, delimiter='\t')
                df_list.append(df)

                idx = 0
                for d in df['happy'].values:
                    if pred_num == 0:
                        happy_pred.append(d)
                    else:
                        happy_pred[idx] += d
                    idx += 1

                idx = 0
                for d in df['sad'].values:
                    if pred_num == 0:
                        sad_pred.append(d)
                    else:
                        sad_pred[idx] += d
                    idx += 1
                
                idx = 0
                for d in df['angry'].values:
                    if pred_num == 0:
                        angry_pred.append(d)
                    else:
                        angry_pred[idx] += d
                    idx += 1

                idx = 0
                for d in df['others'].values:
                    if pred_num == 0:
                        others_pred.append(d)
                    else:
                        others_pred[idx] += d
                    idx += 1
                pred_num += 1 # one time
    
    return df_list, happy_pred, sad_pred, angry_pred, others_pred

df_list, happy_pred, sad_pred, angry_pred, others_pred = voting()
df = df_list[0]

predictions = []
cnt_non_other = 0
cnt = 0 
for i in range(len(happy_pred)):
    pred = np.array([others_pred[i], happy_pred[i], sad_pred[i], angry_pred[i]])
    best_pred = np.argmax(pred)
    predictions.append(label2emotion[best_pred])

    if best_pred != 0:
        cnt_non_other += 1
    else:
        cnt += 1
    
df['label'] = predictions
df['happy'] = happy_pred
df['sad'] = sad_pred
df['angry'] = angry_pred
df['others'] = others_pred

df.to_csv(save_confidence_path, index=None, sep='\t')
print(df)
print("Non Other:",cnt_non_other,"Other:",cnt)

df = df.drop(['others','happy','sad','angry'], axis=1)
df.to_csv(save_prediction_path, index=None, sep='\t')