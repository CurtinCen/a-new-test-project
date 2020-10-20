import inputs
import numpy as np
from sklearn.metrics import f1_score

def show_state_label_speed_range(traffic_data_list):
    collection = []
    for t_data in traffic_data_list:
        ssl_data = t_data.collect_all_speed_state_label()
        collection += ssl_data
    state_dict = {1:[], 2:[], 3:[]}
    for coll in collection:
        state, speed = coll
        state_dict[state].append(speed)
    print("state 1, avg speed %f, max speed %f, min speed %f"%(np.mean(state_dict[1]), np.max(state_dict[1]), np.min(state_dict[1])))
    print("state 2, avg speed %f, max speed %f, min speed %f"%(np.mean(state_dict[2]), np.max(state_dict[2]), np.min(state_dict[2])))
    print("state 3, avg speed %f, max speed %f, min speed %f"%(np.mean(state_dict[3]), np.max(state_dict[3]), np.min(state_dict[3])))


def show_state_label_car_num_range(traffic_data_list):
    collection = []
    for t_data in traffic_data_list:
        ssl_data = t_data.collect_all_speed_state_label()
        collection += ssl_data
    state_dict = {1:[], 2:[], 3:[]}
    for coll in collection:
        state, car_num = coll
        state_dict[state].append(car_num)
    print("state 1, avg car_num %f, max car_num %f, min car_num %f"%(np.mean(state_dict[1]), np.max(state_dict[1]), np.min(state_dict[1])))
    print("state 2, avg car_num %f, max car_num %f, min car_num %f"%(np.mean(state_dict[2]), np.max(state_dict[2]), np.min(state_dict[2])))
    print("state 3, avg car_num %f, max car_num %f, min car_num %f"%(np.mean(state_dict[3]), np.max(state_dict[3]), np.min(state_dict[3])))

def pred_state_using_most_his_state(traffic_data_list):
    pred_list = []
    for traffic_data in traffic_data_list:
        link_id = traffic_data.link_id
        his_state_list = traffic_data.collect_all_his_state_label()
        count = {1:0, 2:0, 3:0}
        for s in his_state_list:
            count[s] += 1
        count = count.items()
        count = sorted(count, key=lambda x:[1])
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
        count = {1:0, 2:0, 3:0}
        for s in his_state_list:
            count[s] += 1
        count = count.items()
        count = sorted(count, key=lambda x:[1])
        pred_state = count[-1][0]
        pred_list.append([link_id, pred_state])
    pred_list = sorted(pred_list, key=lambda x:x[0])
    pred_list = [l[1] for l in pred_list]
    return pred_list

def weighted_f1_score(label_data, pred_data):
    weight_list = []
    for l in label_data:
        if l == 1 or l == 2:
            weight_list.append(0.2)
        elif l == 3:
            weight_list.append(0.6)
        else:
            print("Error in label!")
            sys.exit(0)
    f1 = f1_score(y_true=label_data, y_pred=pred_data, average=None)
    f1 = 0.2*f1[0] + 0.2*f1[1] + 0.6*f1[2]
    return f1


if __name__ == '__main__':
    date_star = 20190701
#load data
    traffic_data_list = inputs.load_data("./traffic/%d.txt"%date_star)
#show some statics
    state_speed_data = inputs.collect_state_speed_from_traffic_data_list(traffic_data_list)
    state_car_num_data = inputs.collect_state_car_num_from_traffic_data_list(traffic_data_list)
    show_state_label_speed_range(state_speed_data)
    show_state_label_car_num_range(state_car_num_data)

#basic baselines
    label_data = inputs.collect_label_data_from_traffic_data_list(traffic_data_list)
    pred_data_his_state = pred_state_using_most_his_state(traffic_data_list)
    pred_data_cur_state = pred_state_using_most_curr_state(traffic_data_list)

    print("his state pred f1 score %f"%(weighted_f1_score(label_data, pred_data_his_state)))
    print("cur state pred f1 score %f"%(weighted_f1_score(label_data, pred_data_cur_state)))






