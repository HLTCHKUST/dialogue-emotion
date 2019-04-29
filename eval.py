from utils.data_reader import prepare_data, prepare_data_loaders
from utils.utils import getMetrics

import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import os

import pandas as pd
import numpy as np
import os 
import math

import random
import numpy as np

from utils import constant

pred_file_path = constant.pred_file_path
ground_file_path = constant.ground_file_path

emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}
label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}

def read_prediction(file_path):
    preds = []
    with open(file_path, "r") as read_file:
        for line in read_file:
            # print(line.replace("\n",""))
            _, _, _, _, label = line.replace("\n", "").split("\t")
            if label in emotion2label:
                preds.append(np.array(emotion2label[label]))
    
    return np.array(preds)

pred = read_prediction(pred_file_path)
one_hot = np.zeros((pred.shape[0], 4))
one_hot[np.arange(pred.shape[0]), pred] = 1
pred = one_hot

ground = read_prediction(ground_file_path)

print(pred, ground)
print(pred.shape, ground.shape)

accuracy, microPrecision, microRecall, microF1 = getMetrics(pred, ground,True)
print(microF1)