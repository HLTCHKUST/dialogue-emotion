from utils import constant
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_classifier(ty="LR"):
    if(ty=="LR"):
        classifier = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    elif(ty=="SVM"):
        classifier = SVC(kernel='rbf')
    return classifier
