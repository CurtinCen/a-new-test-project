
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
                collection.append([h_road[3]])
        return collection

    def collect_all_his_speed(self):
        collection = []
        for his_road_state in self.his_road_state_list:
            for h_road in his_road_state:
                collection.append([h_road[1]])
        return collection

    def collect_all_cur_state_label(self):
        collection = []
        for c_road in self.cur_road_state:
            collection.append([c_road[3]])
        return collection

    def collect_all_cur_speed_label(self):
        collection = []
        for c_road in self.cur_road_state:
            collection.append([c_road[1]])
        return collection

def collect_label_data_from_traffic_data_list(traffic_data_list):
    label_list = []
    for t_data in traffic_data_list:
        label_list.append([t_data.link_id, t_data.pred_label])
    label_list = sorted(label_list, key=lambda x:x[0])
    label_list = [l[1] for l in label_list]
    return label_list

def collect_state_speed_from_traffic_data_list(traffic_data_list):
    state_speed_list = []
    for t_data in traffic_data_list:
        state_speed_list+=t_data.collect_all_speed_state_label()
    return state_speed_list

def collect_state_car_num_from_traffic_data_list(traffic_data_list):
    state_car_num_list = []
    for t_data in traffic_data_list:
        state_car_num_list+=t_data.collect_all_state_car_num_label()
    return state_car_num_list

def load_data(fname):
    with open(fname, 'r') as fin:
        traffic_data_list = []
        context = fin.read()
        lines = context.split('\n')
        for line in lines:
            items = line.split(';')
            link_id, pred_label, cur_time, pred_time = items.split()
            cur_road_state = []
            for c_road in items[1].split():
                c_item = c_road.split(":")
                time_id = int(c_item[0])
                speed, etc_speed, state_label, car_num = c_item[1].split(',')
                cur_road_state.append([time_id, float(speed), float(etc_speed), int(state_label), int(car_num)])

            his_road_state_list = []
            for item in items[2:]:
                his_road_state = []
                for h_road in item.split():
                    h_item = h_road.split(":")
                    time_id = int(h_item[0])
                    speed, etc_speed, state_label, car_num = h_item[1].split(',')
                    his_road_state.append([time_id, float(speed), float(etc_speed), int(state_label), int(car_num)])
                his_road_state_list.append(his_road_state)
            t_data = TrafficData(link_id, pred_label, cur_time, pred_time, cur_road_state, his_road_state_list)
            traffic_data_list.append(t_data)
        return traffic_data_list




