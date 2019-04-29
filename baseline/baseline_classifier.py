from utils import constant
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_classifier(ty="LR", c=1.0, max_depth=5, n_estimators=300, gamma=0):
    if(ty=="LR"):
        classifier = LogisticRegression(solver='lbfgs',multi_class='multinomial', C=c)
    elif(ty=="SVM"):
        classifier = SVC(kernel='linear')
    elif(ty=="XGB"):
        classifier = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, gamma=gamma, n_jobs=4, tree_method="gpu_hist") ## change later ##
    return classifier