import inputs
import numpy as np
from sklearn.metrics import f1_score
import sys

def show_state_label_speed_range(traffic_data_list):
    collection = []
    for t_data in traffic_data_list:
        ssl_data = t_data.collect_all_state_speed_label()
        collection += ssl_data
    state_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}
    for coll in collection:
        state, speed = coll
        state_dict[state].append(speed)
    print("state 0, avg speed %f, max speed %f, min speed %f"%(np.mean(state_dict[0]), np.max(state_dict[0]), np.min(state_dict[0])))
    print("state 1, avg speed %f, max speed %f, min speed %f"%(np.mean(state_dict[1]), np.max(state_dict[1]), np.min(state_dict[1])))
    print("state 2, avg speed %f, max speed %f, min speed %f"%(np.mean(state_dict[2]), np.max(state_dict[2]), np.min(state_dict[2])))
    print("state 3, avg speed %f, max speed %f, min speed %f"%(np.mean(state_dict[3]), np.max(state_dict[3]), np.min(state_dict[3])))
    print("state 4, avg speed %f, max speed %f, min speed %f"%(np.mean(state_dict[4]), np.max(state_dict[4]), np.min(state_dict[4])))


def show_state_label_car_num_range(traffic_data_list):
    collection = []
    for t_data in traffic_data_list:
        ssl_data = t_data.collect_all_state_car_num_label()
        collection += ssl_data
    state_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}
    for coll in collection:
        state, car_num = coll
        state_dict[state].append(car_num)
    print("state 0, avg car_num %f, max car_num %f, min car_num %f"%(np.mean(state_dict[0]), np.max(state_dict[0]), np.min(state_dict[0])))
    print("state 1, avg car_num %f, max car_num %f, min car_num %f"%(np.mean(state_dict[1]), np.max(state_dict[1]), np.min(state_dict[1])))
    print("state 2, avg car_num %f, max car_num %f, min car_num %f"%(np.mean(state_dict[2]), np.max(state_dict[2]), np.min(state_dict[2])))
    print("state 3, avg car_num %f, max car_num %f, min car_num %f"%(np.mean(state_dict[3]), np.max(state_dict[3]), np.min(state_dict[3])))
    print("state 4, avg car_num %f, max car_num %f, min car_num %f"%(np.mean(state_dict[4]), np.max(state_dict[4]), np.min(state_dict[4])))

def pred_with_simple_hy(traffic_data_list):
    pred_list = []
    for traffic_data in traffic_data_list:
        link_id = traffic_data.link_id
        his_state_list = traffic_data.collect_all_his_state_label()
        count = {0:0, 1:0, 2:0, 3:0, 4:0}
        for s in his_state_list:
            count[s] += 1
        #print("state 0 has %d items out of %d"%(count[0], count[0]+ count[1]+count[2]+count[3]))
        if count[2] == 0 and count[3] == 0:
            pred_list.append([link_id, 1])
            continue
        elif count[3] > count[2] and count[3] > count[1] + count[0]:
            pred_list.append([link_id, 3])
        elif count[2] > count[1] + count[0]:
            pred_list.append([link_id, 2])
        else:
            pred_list.append([link_id, 1])

    pred_list = sorted(pred_list, key=lambda x:x[0])
    pred_list = [l[1] for l in pred_list]
    return pred_list


def pred_state_using_most_his_state(traffic_data_list):
    pred_list = []
    for traffic_data in traffic_data_list:
        link_id = traffic_data.link_id
        his_state_list = traffic_data.collect_all_his_state_label()
        count = {0:0, 1:0, 2:0, 3:0, 4:0}
        for s in his_state_list:
            count[s] += 1
        #print("state 0 has %d items out of %d"%(count[0], count[0]+ count[1]+count[2]+count[3]))
        count.pop(0)
        count.pop(4)
        count = count.items()
        count = sorted(count, key=lambda x:x[1])
        pred_state = count[-1][0]
        pred_list.append([link_id, pred_state])
    pred_list = sorted(pred_list, key=lambda x:x[0])
    pred_list = [l[1] for l in pred_list]
    return pred_list

def pred_state_using_most_curr_state(traffic_data_list):
    pred_list = []
    for traffic_data in traffic_data_list:
        link_id = traffic_data.link_id
        cur_state_list = traffic_data.collect_all_cur_state_label()
        count = {0:0, 1:0, 2:0, 3:0, 4:0}
        for s in cur_state_list:
            count[s] += 1
        #print("state 0 has %d items out of %d"%(count[0], count[0]+ count[1]+count[2]+count[3]))
        count.pop(0)
        #count.pop(4)
        count = count.items()
        count = sorted(count, key=lambda x:x[1])
        pred_state = count[-1][0]
        pred_list.append([link_id, pred_state])
    pred_list = sorted(pred_list, key=lambda x:x[0])
    pred_list = [l[1] for l in pred_list]
    return pred_list

def pred_state_using_avg_his_speed(traffic_data_list, speed_range):
    pred_list = []
    for traffic_data in traffic_data_list:
        link_id = traffic_data.link_id
        his_speed_list = traffic_data.collect_all_his_speed()
        avg_speed = np.mean(his_speed_list)
        pred_state = bisect.bisect(speed_range, avg_speed) + 1
        pred_list.append([link_id, pred_state])
    pred_list = sorted(pred_list, key=lambda x:x[0])
    pred_list = [l[1] for l in pred_list]
    return pred_list


def pred_state_using_avg_cur_speed(traffic_data_list, speed_range):
    pred_list = []
    for traffic_data in traffic_data_list:
        link_id = traffic_data.link_id
        his_speed_list = traffic_data.collect_all_cur_speed_label()
        avg_speed = np.mean(his_speed_list)
        pred_state = bisect.bisect(speed_range, avg_speed) + 1
        pred_list.append([link_id, pred_state])
    pred_list = sorted(pred_list, key=lambda x:x[0])
    pred_list = [l[1] for l in pred_list]
    return pred_list

def weighted_f1_score(label_data, pred_data):

    f1 = f1_score(y_true=label_data, y_pred=pred_data, average=None)
    f1 = 0.2*f1[0] + 0.2*f1[1] + 0.6*f1[2] + 0.6*f1[3]
    return f1

def cal_all_state(traffic_data_list):
    state_dict = {}
    for traffic_data in traffic_data_list:
        his_state = traffic_data.collect_all_his_state_label()
        cur_state = traffic_data.collect_all_cur_state_label()
        for s in his_state:
            if s not in state_dict:
                state_dict[s] = 1
        for s in cur_state:
            if s not in state_dict:
                state_dict[s] = 1

    print(state_dict)

def build_upload_data(traffic_data_list, pred_data, fout_name):
    data_list = []
    for traffic_data in traffic_data_list:
        data_list.append(traffic_data.get_upload_format())
    data_list = sorted(data_list, key=lambda x:x[0])
    for i, pred_label in enumerate(pred_data):
        data_list[i].append(pred_label)

    with open(fout_name, 'w') as fout:
        fout.write('link,current_slice_id,future_slice_id,label\n')
        for data in data_list:
            data = [str(d) for d in data]
            out_str = ','.join(data)
            fout.write("%s\n"%out_str)


if __name__ == '__main__':
    date = 20190701
    for i in range(30):
        date_star = date + i

    #label distribution
        date_star = date_star + i
        traffic_data_list = inputs.load_data("./traffic/%s.txt"%str(date_star))
        label_data = inputs.collect_label_data_from_traffic_data_list(traffic_data_list)
        d = {}
        for l in label_data:
            if l not in d:
                d[l] = 0.
            else:
                d[l] += 1
        print(d)
        #sys.exit(0)

        #date_star = 'test'
    #load data
        traffic_data_list = inputs.load_data("./traffic/%s.txt"%str(date_star))
        #cal_all_state(traffic_data_list)
        #sys.exit(0)
    #show some statics
        #state_speed_data = inputs.collect_state_speed_from_traffic_data_list(traffic_data_list)
        #state_car_num_data = inputs.collect_state_car_num_from_traffic_data_list(traffic_data_list)
        #show_state_label_speed_range(traffic_data_list)
        #show_state_label_car_num_range(traffic_data_list)

    #basic baselines
        label_data = inputs.collect_label_data_from_traffic_data_list(traffic_data_list)

        pred_data_his_state = pred_state_using_most_his_state(traffic_data_list)
        #pred_data_cur_state = pred_state_using_most_curr_state(traffic_data_list)
        #pred_data = pred_with_simple_hy(traffic_data_list)

        print("his state pred f1 score %f of dataset %s"%(weighted_f1_score(label_data, pred_data_his_state), str(date_star)))
        #print("cur state pred f1 score %f"%(weighted_f1_score(label_data, pred_data_cur_state)))

        #build_upload_data(traffic_data_list, pred_data_his_state, 'his_result.txt')
        #build_upload_data(traffic_data_list, pred_data_cur_state, 'cur_result.txt')
        #build_upload_data(traffic_data_list, pred_data, 'simple_hy.txt')






