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
