from utils.data_reader import prepare_data_for_feature, generate_vocab, read_data
from utils.utils import getMetrics
from utils.features import get_feature
from utils import constant
from baseline.baseline_classifier import get_classifier
from baseline.baseline_features import get_multi_features

import numpy as np
import argparse

import math
import csv
import os

'''
Try
python main_classifier.py --emb_dim 300
'''
feature_list = [ "emo2vec", "bert", "elmo", "glove-w2v", "deepmoji", "glove-common", "glove-twitter", "emoji",]
## emb_dim for features
## glove-common  -- 300
## glove-twitter -- 200/100/50/25
## glove-w2v     -- 300
## emoji         -- 300/200/100/50
## deepmoji      -- 300
## emo2vec       -- 300
## bert          -- 300
## elmo          -- 300
## define a list recording the possible for each feature
emb_dim_list = [[300],[200,100,50,25],[300,200,100,50]] 
## the second for glove-twitter, the third for emoji, the first for other features

## only need to think about n = 1,2,3,4
## for n = 5,6,7,8, features just like the rest features remaining in n = 3,2,1,0
## for n = 2/6, use select_matrix to decide which features to be selected
## for n = 3/5, use select_matrix and a number(except the indexes of the point in the selected matrix)
## for n = 4,   
## for n = 8,   directly use all the features

## use pre-defined matrix to select 2 features
num_feature = len(feature_list)
select_matrix = np.triu(np.ones(num_feature))
select_matrix = 1 - select_matrix
## find the indexs of elements which value is 1 in the select_matrix
indexs = np.argwhere(select_matrix == 1) ## [28,2] matrix, 2 is the number of features.
## change this index matrix suitable for selecting three features
## add an element which index is larger than any of indexs in the index vector
indexs_for_three = []
for index in indexs:
    s = np.linspace(0,7,8)
    if index[0]<7:
        sub_s = s[index[0]+1:]
        for si in sub_s:
            index_for_three = [int(index[0]), int(index[1]), int(si)]
            index_for_three.sort()
            indexs_for_three.append(index_for_three)
        pass

## predefine variables to evaluate the LR model
F1Max = 0
cMax = 0  # the value of c when F1 score gets max value

## LR model
## use csv file to store the best c for LR model with different features
results = []

## flags for checkpoint
newTry = False   ## one for decision whether use checkpoint
checkPoint = False  ## one for decision whether it's after checkpoint

if newTry:
    print("This a new try for finding the best C for LR model with 8 features")
    pass

if not os.path.exists("baseline/result"):
    os.makedirs("./baseline/result")

for index in indexs_for_three:

    ## find the features to be tested
    features = [] 
    item_head = ""
    for x in index:
        features.append(feature_list[x])
        item_head = item_head + feature_list[x] + "-"
        pass
    
    print("Combining features: {}".format(features))

    ## For loop for embedding dimensions
    ## For only 2 features have different embedding dimensions, only 2 for loops are needed.
    ## Because the 2 features above are place at the end of the feature list, so only the last two items need to be checked
    ## whether it is between the 2 features mentioned above.
    emb_dim = []
    for sf in features[-2:]:
        if sf=="glove-twitter":
            emb_dim.append(emb_dim_list[1])
        elif sf=="emoji":
            emb_dim.append(emb_dim_list[2])
        else:
            emb_dim.append(emb_dim_list[0])
    
    for emb_dim1 in emb_dim[0]: ## could be emoji, glove-twitter or others
        for emb_dim2 in emb_dim[1]: ## could be glove-twiiter or others
            resultMax = []
            F1Max = 0
            cMax = 0
            ## define the format of the filename 
            item =  item_head + str(emb_dim1) + "-" + str(emb_dim2)

            if os.path.exists('baseline/result/LR_multi3_{}_max.csv'.format(str(item))):
                if newTry:
                    with open('baseline/result/LR_multi3_{}_findC.csv'.format(str(item)), 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(["feature","C","microF1"])
                    pass
                    checkPoint = False
                else:
                    with open('baseline/result/LR_multi3_{}_max.csv'.format(str(item)), newline='') as csvfile:
                        mLines = csvfile.readlines()
                        targetLine = mLines[-1]
                        cPoint=targetLine.split(',')[1]
                    checkPoint = True
                    print("for {}, C before {} has been tested".format(str(item),str(cPoint)))
                    pass
            else:
                with open('baseline/result/LR_multi3_{}_max.csv'.format(str(item)), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["feature","C","microF1"])

                with open('baseline/result/LR_multi3_{}_findC.csv'.format(str(item)), 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["feature","C","microF1"])

                checkPoint = False
                pass

            ## compute Micro F1 score for each feature
            for j in range(-5, 3):
                c = math.pow(10, j)

                if (not newTry) and checkPoint:
                    if cPoint=="C":
                        checkPoint = False
                    else:
                        print("{} is skipped because it has been tested!".format(c))
                        if c==float(cPoint):
                            checkPoint = False
                            continue
                        else:
                            continue
                    pass

                model = get_classifier(ty='LR', c=c)

                microF1s = 0
                for i in range(constant.num_split):

                    X_train, y_train, X_test, y_test = get_multi_features(features, i, [emb_dim1,emb_dim2])

                    print("the shape of X and y is {} and {}".format(X_train.shape, y_train.shape))

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

                results.append([item, c, microF1s])
                with open('baseline/result/LR_multi3_{}_max.csv'.format(str(item)), 'a') as f:
                    writer = csv.writer(f)
                    for row in results:
                        writer.writerow(row)
                results = []

                
                if microF1s > F1Max:
                    F1Max = microF1s
                    cMax = c

                resultMax=[item, cMax, F1Max]
                print("(EXPERIMENT %d) Best F1 VAL: %3.5f" % ((i+1), F1Max))

                ## prepare for the next cycle
                microF1s = 0

            if not checkPoint:
                ## write the best result into csv file
                with open('baseline/result/LR_multi3_{}_findC.csv'.format(str(item)), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(resultMax)
                pass
            else:
                print("Didn't find the checkpoint as {}".format(cPoint))
                pass