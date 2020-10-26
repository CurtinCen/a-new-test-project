import sys
import networkx as nx
import os
import pickle as pkl
import numpy as np
from sklearn import preprocessing

class TrafficData():
    def __init__(self, link_id, pred_label=-1, cur_time=None, pred_time=None, cur_road_state=None, his_road_state_list=None):
        self.link_id = link_id
        self.pred_label = pred_label
        self.cur_time = cur_time
        self.pred_time = pred_time
        self.cur_road_state = cur_road_state
        self.his_road_state_list = his_road_state_list

    def collect_all_state_speed_label(self):
        collection = []
        for c_road in self.cur_road_state:
            collection.append([c_road[3], c_road[1]])
        for his_road_state in self.his_road_state_list:
            for h_road in his_road_state:
                collection.append([h_road[3], h_road[1]])
        return collection

    def collect_all_state_car_num_label(self):
        collection = []
        for c_road in self.cur_road_state:
            collection.append([c_road[3], c_road[4]])
        for his_road_state in self.his_road_state_list:
            for h_road in his_road_state:
                collection.append([h_road[3], h_road[4]])
        return collection

    def collect_all_his_state_label(self):
        collection = []
        for his_road_state in self.his_road_state_list:
            for h_road in his_road_state:
                collection.append(h_road[3])
        return collection

    def collect_all_his_speed(self):
        collection = []
        for his_road_state in self.his_road_state_list:
            for h_road in his_road_state:
                collection.append(h_road[1])
        return collection

    def collect_all_cur_state_label(self):
        collection = []
        for c_road in self.cur_road_state:
            collection.append(c_road[3])
        return collection

    def collect_all_cur_speed_label(self):
        collection = []
        for c_road in self.cur_road_state:
            collection.append(c_road[1])
        return collection

    def get_upload_format(self):
        return [self.link_id, self.cur_time, self.pred_time]

def collect_label_data_from_traffic_data_list(traffic_data_list):
    label_list = []
    for t_data in traffic_data_list:
        label_list.append([t_data.link_id, t_data.pred_label])
    label_list = sorted(label_list, key=lambda x:x[0])
    #res_list = [l[1] for l in label_list]
    #replace all 4 with 3
    res_list = []
    for l in label_list:
        if l[1] == 4:
            res_list.append(3)
        else:
            res_list.append(l[1])
    return res_list

def collect_state_speed_from_traffic_data_list(traffic_data_list):
    state_speed_list = []
    for t_data in traffic_data_list:
        state_speed_list+=t_data.collect_all_state_speed_label()
    return state_speed_list

def collect_state_car_num_from_traffic_data_list(traffic_data_list):
    state_car_num_list = []
    for t_data in traffic_data_list:
        state_car_num_list+=t_data.collect_all_state_car_num_label()
    return state_car_num_list

def load_data(fname):
    if os.path.exists('temp/%s.pkl'%fname):
        with open('temp/%s.pkl'%fname, 'rb') as fin:
            traffic_data_list = pkl.load(fin)
    else:
        with open('traffic/%s.txt'%fname, 'r') as fin:
            traffic_data_list = []
            context = fin.read().strip()
            lines = context.split('\n')
            for line in lines:
                items = line.split(';')
                try:
                    link_id, pred_label, cur_time, pred_time = items[0].split()
                    cur_road_state = []
                    for c_road in items[1].split():
                        c_item = c_road.split(":")
                        time_id = int(c_item[0])
                        speed, etc_speed, state_label, car_num = c_item[1].split(',')
                        if int(state_label) == 4:
                            state_label = 3
                        cur_road_state.append([int(time_id), float(speed), float(etc_speed), int(state_label), int(car_num)])
                except ValueError:
                    print(items[0])
                    sys.exit(0)

                his_road_state_list = []
                for item in items[2:]:
                    his_road_state = []
                    for h_road in item.split():
                        h_item = h_road.split(":")
                        time_id = int(h_item[0])
                        speed, etc_speed, state_label, car_num = h_item[1].split(',')
                        if int(state_label) == 4:
                            state_label = 3
                        his_road_state.append([int(time_id), float(speed), float(etc_speed), int(state_label), int(car_num)])
                    his_road_state_list.append(his_road_state)
                if int(pred_label) == 4:
                    pred_label = 3
                t_data = TrafficData(int(link_id), int(pred_label), int(cur_time), int(pred_time), cur_road_state, his_road_state_list)
                traffic_data_list.append(t_data)
        with open('temp/%s.pkl'%fname, 'wb') as fout:
            pkl.dump(traffic_data_list, fout)
    return traffic_data_list





def load_topo(fname):
    with open(fname, 'r') as fin:
        context = fin.read().strip().split('\n')
        adj_dict = {}
        for line in context:
            node, edges = line.split('\t')
            edges = edges.split(',')
            node = int(node)
            edges = [int(e) for e in edges]
            adj_dict[node] = edges
    G = nx.from_dict_of_lists(adj_dict)
    return G

def load_attr(fname):
    if os.path.exists("temp/attr.pkl"):
        with open("temp/attr.pkl", 'rb') as fin:
            X = pkl.load(fin)
            return X
    X = []
    feature2lb = preprocessing.LabelBinarizer()
    feature2lb.fit([1, 2, 3])
    feature3lb = preprocessing.LabelBinarizer()
    feature3lb.fit([1, 2, 3, 4, 5])
    feature4lb = preprocessing.LabelBinarizer()
    feature4lb.fit([2, 3, 4, 5, 6, 7, 8])
    feature5lb = preprocessing.LabelBinarizer()
    feature5lb.fit([1, 2, 3])
    feature7lb = preprocessing.LabelBinarizer()
    feature7lb.fit([1, 2, 3, 4, 5])
    length = []
    direction = []
    pathclass = []
    speedclass = []
    lane_num = []
    speed_limit = []
    level = []
    width = []
    with open(fname, 'r') as fin:
        context = fin.read().strip().split('\n')
        for line in context:
            items = line.split('\t')
            length.append(int(items[1]))
            direction.append(int(items[2]))
            pathclass.append(int(items[3]))
            speedclass.append(int(items[4]))
            lane_num.append(int(items[5]))
            speed_limit.append(float(items[6]))
            level.append(int(items[7]))
            width.append(int(items[8]))

    direction = feature2lb.transform(direction)
    pathclass = feature3lb.transform(pathclass)
    speedclass = feature4lb.transform(speedclass)
    lane_num = feature5lb.transform(lane_num)
    level = feature7lb.transform(level)

    length = np.reshape(np.array(length), (-1, 1))
    speed_limit = np.reshape(np.array(speed_limit), (-1, 1))
    width = np.reshape(np.array(width), (-1, 1))

    X = np.concatenate([length, direction, pathclass, speedclass,lane_num, speed_limit, level, width], axis=1)

    with open("temp/attr.pkl", 'wb') as fout:
        pkl.dump(X, fout)
    return X


#extract raw label data
def extract_raw_label():
    if os.path.exists("temp/labels.pkl"):
        with open("temp/labels.pkl", 'rb') as fin:
            [trainY, valY, testY] = pkl.load(fin)
            return trainY, valY, testY
    #build training data
    date = 20190701
    k = 20
    trainY = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = load_data(str(date_star))
        label_data = collect_label_data_from_traffic_data_list(traffic_data_list)
        trainY += label_data
        print("procee file %s END!!"%str(date_star))

    #build validation
    date = 20190721
    k=5
    valY = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = load_data(str(date_star))
        label_data = collect_label_data_from_traffic_data_list(traffic_data_list)
        valY += label_data
        print("procee file %s END!!"%str(date_star))

    #build test
    date = 20190726
    k=5
    testX = []
    testY = []
    for i in range(k):
        date_star = date + i
        traffic_data_list = load_data(str(date_star))
        label_data = collect_label_data_from_traffic_data_list(traffic_data_list)
        testY += label_data
        print("procee file %s END!!"%str(date_star))
    with open('temp/labels.pkl', 'wb') as fout:
        pkl.dump([trainY, valY, testY], fout)
    return trainY, valY, testY


