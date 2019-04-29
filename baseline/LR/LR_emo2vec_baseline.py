from utils.data_reader import prepare_data_for_feature, generate_vocab, read_data
from utils.utils import getMetrics
from utils.features import get_feature
from utils import constant
from baseline.baseline_classifier import get_classifier
import numpy as np
import argparse

import math
import csv
import os

'''
Try
python main_classifier.py --emb_dim 300
'''
# feature_list = ['glove-common','glove-twitter','glove-w2v','emoji','elmo','bert','deepmoji','emo2vec']
# feature_list = ["glove-common", "emoji-300", "deepmoji", "elmo", "bert", "emo2vec"]
feature_list = ["emo2vec"]
for item in feature_list:
    filename = str(item)+'-'+str(constant.emb_dim)
    ## predefine variables to evaluate the LR model
    F1Max = 0
    cMax = 0  # the value of c when F1 score gets max value

    ## LR model
    ## use csv file to store the best c for LR model with different features
    results = []

    if not os.path.exists("baseline/result"):
        os.makedirs("./baseline/result")

    with open('baseline/result/LR_single_{}_max.csv'.format(filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["feature","C","microF1"])

    with open('baseline/result/LR_single_{}_findC.csv'.format(filename), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["feature","C","microF1"])

    # data_loaders_tr, data_loaders_val, data_loaders_test, vocab = prepare_data_loaders(num_split=constant.num_split, batch_size=constant.batch_size, hier=False)

    ## distinguish twitter glove and common glove
    if item[:5]=='glove':
        ty = item[6:]
        feature = item[:5]
    elif item.find('-')>0:
        ty = 'common'
        feature = item[:(item.find('-')-1)]
    else:
        ty = 'common'
        feature = item

    ## compute Micro F1 score for each feature
    for j in range(2, 24):
        c = j/2000
        model = get_classifier(ty='LR', c=c)

        microF1s = 0
        for i in range(constant.num_split):

            ## prepare data for feature-10 folders
            vocab = generate_vocab()
            train, val, dev_no_lab = read_data(is_shuffle=True, random_state=i)
            ## feature_list: glove emoji elmo bert deepmoji emo2vec
            ## if you want twitter glove or common glove use  ty='twitter' and ty='common'
            X_train, y_train = get_feature(train, vocab, feature_list=[feature], mode=['sum'],split="train",ty=ty) ## [29010,3,emb_size] 3 is number of sentence
            X_test, y_test = get_feature(val, vocab, feature_list=[feature], mode=['sum'],split="valid",ty=ty) ## [1150,3,emb_size]

            print("###### EXPERIMENT %d when C equals to %f ######" % ((i+1),c))
            print("(EXPERIMENT %d) Create the model" % (i+1))

            ## train aval and predict
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train) ## [29010,3,emb_size] --> [29010, 3 * emb_size]

            ## validate to validation set 
            y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

            ## covert output to one hot
            one_hot = np.zeros((y_pred.shape[0], 4))
            one_hot[np.arange(y_pred.shape[0]), y_pred] = 1
            ## call the scorer 
            acc, microPrecision, microRecall, microF1 = getMetrics(one_hot,y_test,verbose=True)

            microF1s = microF1s + microF1
        
        microF1s = microF1s/constant.num_split

        results.append([filename, c, microF1s])
        with open('baseline/result/LR_single_{}_max.csv'.format(filename), 'a') as f:
            writer = csv.writer(f)
            for row in results:
                writer.writerow(row)
        results = []

        
        if microF1s > F1Max:
            F1Max = microF1s
            cMax = c

        resultMax=[filename, cMax, F1Max]
        print("(EXPERIMENT %d) Best F1 VAL: %3.5f" % ((i+1), F1Max))

        ## prepare for the next cycle
        microF1s = 0

    ## write the best result into csv file
    with open('baseline/result/LR_single_{}_findC.csv'.format(filename), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(resultMax)

    resultMax = []
    F1Max = 0