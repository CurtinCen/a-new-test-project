from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from inputs import load_data
import inputs

from models import Classifier, weighted_f1_score
import pickle as pkl

#data partitioning, 0701~0720 training, 0721~0725 validation, 0726~0730 test

def extract_raw_features(traffic_data_list):
    X = []
    for traffic_data in traffic_data_list:
        features = [traffic_data.cur_time, traffic_data.pred_time]
        for c_road in traffic_data.cur_road_state:
            features += c_road
        for his_road_state in traffic_data.his_road_state_list:
            for h_road in his_road_state:
                features += h_road
        X.append((traffic_data.link_id, features))
    X = sorted(X, key=lambda x:x[0])
    X = [f[1] for f in X]
    return X


#simple linear model, load all raw features
def raw_features():
    #build training data
    date = 20190701
    k = 20
    trainX = []
    trainY = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("./traffic/%s.txt"%str(date_star))
        label_data = inputs.collect_label_data_from_traffic_data_list(traffic_data_list)
        features = extract_raw_features(traffic_data_list)
        trainX += features
        trainY += label_data

    #build validation
    date = 20190721
    k=5
    valX = []
    valY = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("./traffic/%s.txt"%str(date_star))
        label_data = inputs.collect_label_data_from_traffic_data_list(traffic_data_list)
        features = extract_raw_features(traffic_data_list)
        valX += features
        valY += label_data

    #build test
    date = 20190726
    k=5
    testX = []
    testY = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("./traffic/%s.txt"%str(date_star))
        label_data = inputs.collect_label_data_from_traffic_data_list(traffic_data_list)
        features = extract_raw_features(traffic_data_list)
        testX += features
        testY += label_data
    #pass
    return trainX, trainY, valX, valY, testX, testY


if __name__ == '__main__':
    if os.path.exists("temp/raw_features.pkl"):
        with open("temp/raw_features.pkl", 'rb') as fin:
            [trainX, trainY, valX, valY, testX, testY] = pkl.load(fin)
    else:
        trainX, trainY, valX, valY, testX, testY = raw_features()
        with open("temp/raw_features.pkl", 'wb') as fout:
            pkl.dump([trainX, trainY, valX, valY, testX, testY], fout)

    class_num = 3
    clf = Classifier('LR', class_num)
    clf.train(trainX, trainY)
    pred_trainX = clf.pred(trainX)
    pred_valX = clf.pred(valX)

    print("f1-score in training %f, in validation %f"%(weighted_f1_score(trainY, pred_trainX), weighted_f1_score(valY, pred_valX)))





