from utils.data_reader import prepare_data_for_feature
from utils.utils import getMetrics
from utils.features import get_feature
from utils import constant
from models.classifier import get_classifier
import numpy as np

'''
Try
python main_classifier.py --emb_dim 300
'''

train, val, dev_no_lab, vocab = prepare_data_for_feature()
## feature_list: glove emoji elmo bert deepmoji emo2vec
## if you want twitter glove or common glove use  ty='twitter' and ty='common'
X_train, y_train = get_feature(train, vocab, feature_list=['glove'], mode=['sum'],split="train",ty='common') ## [29010,3,emb_size] 3 is number of sentence
X_test, y_test = get_feature(val, vocab, feature_list=['glove'], mode=['sum'],split="valid",ty='common') ## [1150,3,emb_size]

model = get_classifier(ty='LR')

## train aval and predict
model.fit(X_train.reshape(X_train.shape[0], -1), y_train) ## [29010,3,emb_size] --> [29010, 3 * emb_size]

## validate to validation set 
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

## covert output to one hot
one_hot = np.zeros((y_pred.shape[0], 4))
one_hot[np.arange(y_pred.shape[0]), y_pred] = 1
## call the scorer 
getMetrics(one_hot,y_test,verbose=True)