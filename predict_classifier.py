from utils.data_reader import prepare_data_for_feature, generate_vocab, read_data
from utils.features import get_feature
from utils.utils import getMetrics
from utils import constant
from baseline.baseline_classifier import get_classifier
from baseline.baseline_features import get_features_for_prediction
import numpy as np
import csv
import pandas as pd
import os
'''
Before running this file, pls assign save path. 
python predict_classifier.py --save_path 'save/LR_final/' --classifier 'LR' --C 0.01 --pred_score --include_test
'''
if not os.path.exists(constant.save_path):
    os.makedirs(constant.save_path)

label2emotion = ["others","happy", "sad","angry"]
## define parameters for getting feature
features = constant.features

## define parameters for building model
classifier_list = ["LR","SVM","XGB"]
## LR: c
## SVM: c
## XGB: n_estimators, max_depth
parameter_list = [constant.C,constant.n_estimators,constant.max_depth]
classifier = constant.classifier

print('features: ', features)
print('Classifier: ', classifier)
print('Parameters: ', parameter_list)

txt_file = classifier+"_baseline.txt"
microF1s = 0

## define parameters for checkpoint
if classifier=="XGB":
    params = str(parameter_list[1])+"-"+str(parameter_list[2])
    pass
else:
    params = str(parameter_list[0])
    pass
record_file = classifier+"_"+params+".csv"

checkpoint = False
currentSplit = 0

## check checkpoint
if os.path.exists(constant.save_path+record_file):
    checkpoint = True
    ## read checkpoint
    with open(constant.save_path+record_file, newline='') as csvfile:
        mLines = csvfile.readlines()
        ## get current split
        targetLine = mLines[-1]
        currentSplit=targetLine.split(',')[0]
        ##read F1 score records
        rLines = mLines[-currentSplit-1:]
        for line in rLines:
            microF1s += float(line.split(',')[1])
    currentSplit += 1

model = get_classifier(ty=classifier, c=parameter_list[0], n_estimators=parameter_list[1], max_depth=parameter_list[2])

for i in range(constant.num_split):
    ## confirm checkpoint
    if checkpoint==True and i<currentSplit:
        print("Split {} is skipped because it has been run!".format(i))
        continue

    ## prepare feature for model
    X_train, y_train, X_val, y_val, X_test, ind, X_text = get_features_for_prediction(features, i, use_pca=False)
    print('shape of X_train',X_train.shape)
    print('shape of X_test',X_test.shape)
    print("###### Running folder %d ######" % (i+1))

    if i==0:
        y_pred = []
        pass
    ## train aval and predict
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train) ## [29010,3,emb_size] --> [29010, 3 * emb_size]

    ## validate to validation set 
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

    print("###### Writing result of folder %d to file ######" % (i+1))

    ## generate files with 3 turns and labels
    file = constant.save_path+"test_{}.txt".format(i)

    if not os.path.exists(file):
        with open(file, 'w') as the_file:
            the_file.write("id\tturn1\tturn2\tturn3\tlabel\n")
            preds_dict = {}
            indices = []
            for idx, text, pred in zip(ind,X_text,y_pred):
                preds_dict[idx] = "{}\t{}\t{}\t{}\t{}\n".format(idx,text[0],text[1],text[2],label2emotion[pred])
                indices.append(idx)

            sorted_indices = np.argsort(-np.array(indices))[::-1]
            for idx in range(len(sorted_indices)):
                the_file.write(preds_dict[idx])
    
    ## run validation set to get the F1 score
    if constant.pred_score:
        if i==0:
            txtfile = open(txt_file,'a')
            txtfile.write("\n--------------------\n")
            txtfile.write("Classifier %s, Parameters: %f, %f, %f" %(classifier, parameter_list[0], parameter_list[1], parameter_list[2]))
            txtfile.close()

        y_pred_val = model.predict(X_val.reshape(X_val.shape[0], -1))

        ## covert output to one hot
        one_hot = np.zeros((y_pred_val.shape[0], 4))
        one_hot[np.arange(y_pred_val.shape[0]), y_pred_val] = 1
        ## call the scorer 
        acc, microPrecision, microRecall, microF1 = getMetrics(one_hot,y_val,verbose=True)

        txtfile = open(txt_file,'a')
        txtfile.write("(EXPERIMENT %d) microF1 score %f" % ((i+1), microF1))
        txtfile.write("\n--------------------\n")
        txtfile.close()

        result = [i,microF1]
        with open(constant.save_path+record_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(result)

        microF1s = microF1s + microF1
    
microF1s = microF1s/constant.num_split
txtfile = open(txt_file,'a')
txtfile.write("\nAVERAGE F1 VAL: %3.5f\n\n" % microF1s)
txtfile.close()
    