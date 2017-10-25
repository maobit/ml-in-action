#!/usr/bin/env python

import math


def calc_entropy(data_set):
    num_entries = len(data_set)
    label_cnts = {}
    for feat_vec in data_set:
        cur_label = feat_vec[-1]
        label_cnts[cur_label] = label_cnts.get(cur_label, 0) + 1
    entropy = 0.0
    for key in label_cnts:
        prob = float(label_cnts[key]) / num_entries
        entropy -= prob * math.log(prob, 2)
    return entropy


def create_data():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feat_to_split(data_set):
    # 选择最好的分割特征
    num_feats = len(data_set[0]) - 1
    base_entropy = calc_entropy(data_set)
    best_info_gain = 0.0
    best_feat = -1
    for i in range(num_feats):
        feat_list = [x[i] for x in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0

        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = i

    return best_feat


def majority_cnt(class_list):
    # 多数投票
    import operator
    class_cnt = {}
    for vote in class_list:
        class_cnt[vote] = class_cnt.get(vote, 0) + 1
    sorted_class_cnt = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_cnt[0][0]


def create_tree(data_set, labels):
    # 递归地构建决策树
    class_list = [x[-1] for x in data_set]
    if len(set(class_list)) == 1:  # 类别完全相同则停止继续划分
        return class_list[0]
    if len(data_set[0]) == 1:  # 遍历完所有的特征时，返回出现次数最多的类别
        return majority_cnt(class_list)
    best_feat = choose_best_feat_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    feat_values = [x[best_feat] for x in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, filename):
    import pickle
    with open(filename, 'w') as f:
        pickle.dump(input_tree, f)


def load_tree(filename):
    import pickle
    with open(filename) as f:
        return pickle.load(f)

