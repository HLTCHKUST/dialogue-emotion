from utils.data_reader import prepare_data_for_feature, generate_vocab, read_data
from utils.features import get_feature
from sklearn.decomposition import PCA

from utils import constant
import numpy as np
import argparse

def featureAnalysis(item):
    if item[:5]=='glove':
        ty = item[6:]
        mode = 'sum'
        feature = item[:5]
    elif item[:8]=='deepmoji':
        ty = 'common'
        mode = item[9:]
        feature = item[:8]
    elif item.find('-')>0:
        ty = 'common'
        mode = 'sum'
        feature = item[:(item.find('-')-1)]
    else:
        ty = 'common'
        mode = 'sum'
        feature = item
    return feature, ty, mode

def get_multi_features(features, i, emb_dim, use_pca=False):
    X_train, y_train, X_test, y_test = [],[],[],[]

    for item in features:
        ## distinguish twitter glove and common glove
        ## distinguish deepmoji sum and avg
        feature, ty, mode = featureAnalysis(item)
        
        if item == features[-2]:
            constant.emb_dim = emb_dim[0]
        elif item == features[-1]:
            constant.emb_dim = emb_dim[1]
        else:
            constant.emb_dim = 300

        print(feature)
        ## prepare data for feature-10 folders
        vocab = generate_vocab()
        train, val, dev_no_lab = read_data(is_shuffle=True, random_state=i, dev_with_label=constant.dev_with_label, include_test=constant.include_test)

        ## feature_list: glove emoji elmo bert deepmoji emo2vec
        ## if you want twitter glove or common glove use  ty='twitter' and ty='common'
        split_train = "merged_train"+str(i) if constant.include_test else "train"+str(i)
        split_val = "merged_val"+str(i) if constant.include_test else "valid"+str(i)

        print("Loading split", split_train)

        Xi_train, yi_train = get_feature(train, vocab, feature_list=[feature], mode=[mode],split=split_train,ty=ty) ## [29010,3,emb_size] 3 is number of sentence
        Xi_test, yi_test = get_feature(val, vocab, feature_list=[feature], mode=[mode],split=split_val,ty=ty) ## [1150,3,emb_size]

        if use_pca:
            Xi_train, Xi_test, _ = pca(Xi_train, Xi_test)
            pass

        if feature == "bert":
            Xi_train = np.squeeze(Xi_train,axis = 2)
            Xi_test = np.squeeze(Xi_test,axis = 2)
            pass
        if X_train==[]:
            X_train = Xi_train
            y_train = yi_train
            X_test = Xi_test
            y_test = yi_test
        else:
            X_train = np.concatenate((X_train, Xi_train), axis = 2)
            X_test = np.concatenate((X_test, Xi_test), axis = 2)
    return X_train, y_train, X_test, y_test

def pca(X_train, X_val, X_test=np.zeros([1,3,1])):
    trainShape = X_train.shape
    valShape = X_val.shape
    testShape = X_test.shape
    
    ## Use PCA to decrease the dimension of features to accelerate SVM
    # pca_dim = int(X_train.shape[2]/3)
    pca = PCA(n_components=0.9)

    X_train_reduced = pca.fit_transform(X_train.reshape(trainShape[0]*trainShape[1], trainShape[2])) ## [29010,3,emb_size] --> [29010 * 3, emb_size]
    X_val_reduced = pca.transform(X_val.reshape(valShape[0]*valShape[1], valShape[2]))

    X_train_reduced = X_train_reduced.reshape(trainShape[0],trainShape[1],-1)  ## [29010 * 3, pca_dim] --> [29010, 3, pca_dim]
    X_val_reduced = X_val_reduced.reshape(valShape[0],valShape[1],-1)

    if not (X_test==np.zeros([1,3,1])).all():
        print("Generating X_test without labels")
        X_test_reduced = pca.transform(X_test.reshape(testShape[0]*testShape[1], testShape[2]))
        X_test_reduced = X_test_reduced.reshape(testShape[0],testShape[1],-1)
        pass
    else:
        X_test_reduced = X_val_reduced

    print("The shape of X after pca is {}".format(X_train_reduced.shape))
    return X_train_reduced, X_val_reduced, X_test_reduced

def get_single_feature_for_svm(feature, ty, i):
    
    ## prepare data for feature-10 folders
    vocab = generate_vocab()
    train, val, dev_no_lab = read_data(is_shuffle=True, random_state=i, dev_with_label=constant.dev_with_label, include_test=constant.include_test)
    ## feature_list: glove emoji elmo bert deepmoji emo2vec
    ## if you want twitter glove or common glove use  ty='twitter' and ty='common'
    X_train, y_train = get_feature(train, vocab, feature_list=[feature], mode=['sum'],split="train",ty=ty) ## [29010,3,emb_size] 3 is number of sentence
    X_test, y_test = get_feature(val, vocab, feature_list=[feature], mode=['sum'],split="valid",ty=ty) ## [1150,3,emb_size]
    
    X_train_reduced, X_test_reduced, _ = pca(X_train, X_test)

    return X_train_reduced, y_train, X_test_reduced, y_test

def get_features_for_prediction(features, i, use_pca=False):
    X_train, y_train, X_test, X_val, y_val = [],[],[],[],[]

    for item in features:
        ## distinguish twitter glove and common glove
        ## distinguish deepmoji sum and avg
        feature, ty, mode = featureAnalysis(item)

        if feature=="glove" and ty=="twitter":
            constant.emb_dim = 200
        elif: feature=="emoji":
            pass
        else:
            constant.emb_dim = 300
            pass
        pass

        print(feature)
        ## prepare data for feature-10 folders
        vocab = generate_vocab(include_test=True)
        train, val, dev_no_lab = read_data(is_shuffle=True, random_state=i, dev_with_label=False, include_test=True)
        ## Add labels to dev_no_lab for getting features
        ind = dev_no_lab[0]
        X_text = dev_no_lab[1]
        labels = ["others" for i in range(len(ind))]
        dev = (ind, X_text, labels)
        
        ## feature_list: glove emoji elmo bert deepmoji emo2vec
        ## if you want twitter glove or common glove use  ty='twitter' and ty='common'
        print(ty)
        Xi_train, yi_train = get_feature(train, vocab, feature_list=[feature], mode=[mode],split="final_train"+str(i),ty=[ty]) ## [29010,3,emb_size] 3 is number of sentence
        # Xi_val, yi_val = get_feature(val, vocab, feature_list=[feature], mode=[mode],split="final_valid"+str(i),ty=[ty]) ## [1150,3,emb_size]        
        Xi_test, _ = get_feature(dev, vocab, feature_list=[feature], mode=[mode],split="final_test"+str(i),ty=[ty]) ## [2755,3,emb_size]

        # Xi_train = np.concatenate((Xi_train, Xi_val), axis = 0)
        # yi_train = np.concatenate((yi_train, yi_val), axis = 0)
        if use_pca:
            Xi_train, Xi_val, Xi_test = pca(Xi_train,Xi_val,Xi_test)
            pass

        # if feature == "bert":
        #     print(Xi_train.shape)
        #     Xi_train = np.squeeze(Xi_train,axis = 2)
        #     Xi_test = np.squeeze(Xi_test,axis = 2)
        #     Xi_val = np.squeeze(Xi_val,axis = 2)
        #     pass
        if X_train==[]:
            X_train = Xi_train
            y_train = yi_train
            X_test = Xi_test
            # X_val = Xi_val
            # y_val = yi_val
        else:
            X_train = np.concatenate((X_train, Xi_train), axis = 2)
            X_test = np.concatenate((X_test, Xi_test), axis = 2)
            # X_val = np.concatenate((X_val, Xi_val), axis = 2)
    return X_train, y_train, X_val, y_val, X_test, ind, X_text