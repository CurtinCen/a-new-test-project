from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np




def weighted_f1_score(label_data, pred_data):

    f1 = f1_score(y_true=label_data, y_pred=pred_data, average=None)
    f1 = 0.2*f1[0] + 0.2*f1[1] + 0.6*f1[2]
    return f1


class Classifier():

    def __init__(self, clf_model, class_num):
        self.clf = self._get_clf_model(clf_model)
        self.class_num = class_num

    def _get_clf_model(self, clf_model):
        if clf_model == 'LR':
            clf = LR()
            params = {'penalty':'l2', 'dual':False, 'tol':0.0001, 'C':0.5, 'fit_intercept':True, 'intercept_scaling':1,
              'class_weight':None, 'random_state':None, 'solver':'lbfgs', 'max_iter':1000, 'multi_class':'multinomial',
              'verbose':0, 'warm_start':False, 'n_jobs':1}
        elif clf_model == 'SVM':
            clf = SVC()
            params = {'C':1.0, 'kernel':'linear', 'degree':3, 'gamma':'auto', 'coef0':0.0, 'shrinking':True,
                  'probability':True, 'tol':0.001, 'cache_size':200, 'class_weight':None, 'verbose':False,
                  'max_iter':-1, 'decision_function_shape':'ovr', 'random_state':None}
        else:
        	print("No such classification model")
        	sys.exit(-1)


        clf.set_params(**params)
        clf = OneVsRestClassifier(clf)
        return clf


    def train(self, X, Y):
        self.clf.fit(X, Y)


    def pred(self, X):
        return self.clf.predict(X)