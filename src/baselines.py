from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from inputs import load_data
import inputs
import sys, time, os
import numpy as np

from models import Classifier, weighted_f1_score
import pickle as pkl

#data partitioning, 0701~0720 training, 0721~0725 validation, 0726~0730 test

def extract_statistical_features():
    #feature 1: [mean history speed, std h speed, max h speed, min h speed]
    #feature 2: [mean history car number, std, max, min]
    if os.path.exists('temp/sta_feature_dict.pkl'):
        with open('temp/sta_feature_dict.pkl', 'rb') as fin:
            sta_feature_dict = pkl.load(fin)
            return sta_feature_dict

    features_dict = {}
    #build training data
    date = 20190701
    k = 20
    features_dict = {}
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        for traffic_data in traffic_data_list:
            link_id = traffic_data.link_id
            if link_id not in features_dict:
                features_dict[link_id] = {}
                features_dict[link_id]['h_speed'] = []
                features_dict[link_id]['car_num'] = []
            for his_road_state in traffic_data.his_road_state_list:
                for h_road in his_road_state:
                    h_speed = h_road[1]
                    car_num = h_road[4]
                    features_dict[link_id]['h_speed'].append(h_speed)
                    features_dict[link_id]['car_num'].append(car_num)
    sta_feature_dict = {}
    for link_id in features_dict:
        mean_speed = np.mean(features_dict[link_id]['h_speed'])
        max_speed = np.max(features_dict[link_id]['h_speed'])
        min_speed = np.min(features_dict[link_id]['h_speed'])
        std_speed = np.std(features_dict[link_id]['h_speed'])

        mean_car_num = np.mean(features_dict[link_id]['car_num'])
        max_car_num = np.max(features_dict[link_id]['car_num'])
        min_car_num = np.min(features_dict[link_id]['car_num'])
        std_car_num = np.std(features_dict[link_id]['car_num'])

        sta_feature_dict[link_id] = [mean_speed, std_speed, max_speed, min_speed, mean_car_num, std_car_num, max_car_num, min_car_num]
    with open('temp/sta_feature_dict.pkl', 'wb') as fout:
        pkl.dump(sta_feature_dict, fout)
    return sta_feature_dict


def extract_raw_features_func(traffic_data_list):
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

def extract_sta_features_func(traffic_data_list, sta_feature_dict):
    if os.path.exists("temp/mean_sta_features.pkl"):
        with open("temp/mean_sta_features.pkl", 'rb') as fin:
            mean_sta_feature_vec = pkl.load(fin)
    else:
        features = np.array(list(sta_feature_dict.values()))
        mean_sta_feature_vec = np.mean(features, axis=0)
        with open("temp/mean_sta_features.pkl", 'wb') as fout:
            pkl.dump(mean_sta_feature_vec, fout)

    features = []
    for traffic_data in traffic_data_list:
        link_id = traffic_data.link_id
        if link_id in sta_feature_dict:
            features.append((link_id, sta_feature_dict[link_id]))
        else:
            features.append((link_id, mean_sta_feature_vec))
    features = sorted(features, key=lambda x:x[0])
    features = [f[1] for f in features]
    return features

def extract_nb_feature_func(traffic_data_list, sta_feature_dict):
    features = []
    for traffic_data in traffic_data_list:
        link_id = traffic_data.link_id
        if not graph.has_node(link_id):
            print("link %d not in graph"%link_id)
            features.append((link_id, [0., 0., 0., 0., 0., 0., 0., 0.]))
        else:
            nbs = graph.neighbors(link_id)
            nb_item_list = []
            for n in nbs:
                if n in sta_feature_dict:
                    t = sta_feature_dict[n]
                else:
                    t = mean_sta_feature_vec
                nb_item_list.append(t)
            nb_item_list = np.array(nb_item_list)
            nb_item_mean = np.mean(nb_item_list, axis=0)
            nb_item_std = np.std(nb_item_list, axis=0)
            nb_item_max = np.max(nb_item_list, axis=0)
            nb_item_min = np.min(nb_item_list, axis=0)
            nb_f = nb_item_mean.tolist() + nb_item_std.tolist() + nb_item_max.tolist() + nb_item_min.tolist()
            features.append((link_id, nb_f))
    features = sorted(features, key=lambda x:x[0])
    features = [f[1] for f in features]
    return features

def extract_raw_attr_func(traffic_data_list, link_attr):
    features = []
    id_limit = len(link_attr)
    feature_limit = len(link_attr[0])
    padding_feature = np.zeros(feature_limit)
    for traffic_data in traffic_data_list:
        link_id = traffic_data.link_id
        if link_id >= id_limit:
            features.append((link_id, padding_feature))
        else:
            features.append((link_id, link_attr[link_id]))
    features = sorted(features, key=lambda x:x[0])
    features = [f[1] for f in features]
    features = np.concatenate(features, axis=0)
    return features

def build_test_X(traffic_data_list, feature_flag):
    X = []
    if feature_flag[0]:
        f1 = extract_raw_features_func(traffic_data_list)
        X.append(f1)
    if feature_flag[1]:
        sta_feature_dict = extract_statistical_features()
        f2 = extract_sta_features_func(traffic_data_list, sta_feature_dict)
        X.append(f2)
    if feature_flag[2]:
        sta_feature_dict = extract_statistical_features()
        f3 = extract_nb_feature_func(traffic_data_list, sta_feature_dict)
        X.append(f3)
    if feature_flag[3]:
        link_attr = inputs.load_attr('traffic/attr.txt')
        f4 = extract_raw_attr_func(traffic_data_list, link_attr)
        X.append(f4)
    X = np.concatenate(X, axis=1)
    return X



#extract statistical feature and build train data format
def sta_features(sta_feature_dict):
    if os.path.exists("temp/sta_features.pkl"):
        with open("temp/sta_features.pkl", 'rb') as fin:
            [trainX, valX, testX] = pkl.load(fin)
            return trainX, valX, testX

    

    #build training data
    date = 20190701
    k = 20
    trainX = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        features = extract_sta_features_func(traffic_data_list, sta_feature_dict)
        trainX += features
        print("procee file %s END!!"%str(date_star))

    #build validation
    date = 20190721
    k=5
    valX = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        features = extract_sta_features_func(traffic_data_list, sta_feature_dict)    
        valX += features
        print("procee file %s END!!"%str(date_star))

    #build test
    date = 20190726
    k=5
    testX = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        features = extract_sta_features_func(traffic_data_list, sta_feature_dict)
        testX += features
        print("procee file %s END!!"%str(date_star))

    with open("temp/sta_features.pkl", 'wb') as fout:
        pkl.dump([trainX, valX, testX], fout)
    return trainX, valX, testX



def extract_features1(sta_feature_dict):
    #extract neighbor sta features
    #feature 1: [mean history speed of all neighbors, std h speed, max h speed, min h speed]
    #feature 2: [mean history car number of all neighbors, std, max, min]

    if os.path.exists("temp/nb_sta_features.pkl"):
        with open("temp/nb_sta_features.pkl", 'rb') as fin:
            [trainX, valX, testX] = pkl.load(fin)
            return trainX, valX, testX

    topo_file = 'traffic/topo.txt'
    graph = inputs.load_topo(topo_file)

    if os.path.exists("temp/mean_sta_features.pkl"):
        with open("temp/mean_sta_features.pkl", 'rb') as fin:
            mean_sta_feature_vec = pkl.load(fin)
    else:
        features = np.array(sta_feature_dict.values())
        mean_sta_feature_vec = np.mean(features, axis=0)

    #build training data
    date = 20190701
    k = 20
    trainX = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        features = extract_nb_feature_func(traffic_data_list, sta_feature_dict)
        trainX += features
        print("procee file %s END!!"%str(date_star))

    #build validation
    date = 20190721
    k=5
    valX = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        features = extract_nb_feature_func(traffic_data_list, sta_feature_dict)
        valX += features
        print("procee file %s END!!"%str(date_star))

    #build test
    date = 20190726
    k=5
    testX = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        features = extract_nb_feature_func(traffic_data_list, sta_feature_dict)
        testX += features
        print("procee file %s END!!"%str(date_star))

    with open("temp/nb_sta_features.pkl", 'wb') as fout:
        pkl.dump([trainX, valX, testX], fout)
    return trainX, valX, testX




#simple linear model, load all raw features
def raw_features(feature_name='raw_features', extract_func=extract_raw_features_func):
    if os.path.exists("temp/%s.pkl"%feature_name):
        with open("temp/%s.pkl"%feature_name, 'rb') as fin:
            [trainX, valX, testX] = pkl.load(fin)
            return trainX, valX, testX


    #build training data
    date = 20190701
    k = 20
    trainX = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        features = extract_func(traffic_data_list)
        trainX += features
        print("procee file %s END!!"%str(date_star))

    #build validation
    date = 20190721
    k=5
    valX = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        features = extract_func(traffic_data_list)
        valX += features
        print("procee file %s END!!"%str(date_star))

    #build test
    date = 20190726
    k=5
    testX = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = inputs.load_data("%s"%str(date_star))
        features = extract_func(traffic_data_list)
        testX += features
        print("procee file %s END!!"%str(date_star))

    with open("temp/%s.pkl"%feature_name, 'wb') as fout:
        pkl.dump([trainX, valX, testX], fout)
    return trainX, valX, testX




if __name__ == '__main__':
    start_time = time.time()
    #extract raw without preprocess
    #trainX0, valX0, testX0 = raw_features('raw_features', extract_raw_features_func)
    #print("process raw features end!")

    #extract statistical features
    sta_feature_dict = extract_statistical_features()
    print("build sta features dict end!")
    trainX1, valX1, testX1 = sta_features(sta_feature_dict)
    print("process statistical features end!")

    #trainX2, valX2, testX2 = extract_features1(sta_feature_dict)
    #print("process neighbor sta features end!")

    #trainX = trainX0
    #valX = valX0
    #testX = testX0

    trainX = trainX1
    valX = valX1
    testX = testX1

    #trainX = np.concatenate((trainX0, trainX1), axis=1)
    #valX = np.concatenate((valX0, valX1), axis=1)
    #testX = np.concatenate((testX0, testX1), axis=1)

    #print("load feature data totally costs %f seconds"%(time.time()-start_time))

    #start_time = time.time()
    #link_attr = inputs.load_attr('traffic/attr.txt')
    #print("load link attributes totally costs %f seconds"%(time.time()- start_time))


    start_time = time.time()
    trainY, valY, testY = inputs.extract_raw_label()
    print("load label data totally costs %f seconds"%(time.time()-start_time))


    start_time = time.time()
    class_num = 3
    clf = Classifier('LR', class_num)
    clf.train(trainX, trainY)
    print("training model totally costs %f seconds"%(time.time()-start_time))

    start_time = time.time()
    pred_trainX = clf.pred(trainX)
    print("pred train data totally costs %f seconds"%(time.time()-start_time))

    start_time = time.time()
    pred_valX = clf.pred(valX)
    print("pred val data totally costs %f seconds"%(time.time()-start_time))

    print("f1-score in training %f, in validation %f"%(weighted_f1_score(trainY, pred_trainX), weighted_f1_score(valY, pred_valX)))


    feature_flag = [0, 0, 0, 0]
    test_traffic_data_list = load_data('test')
    testX = build_test_X(test_traffic_data_list, feature_flag)
    testY = clf.pred(testX)
    build_upload_data(test_traffic_data_list, testY, 'upload1.csv')






