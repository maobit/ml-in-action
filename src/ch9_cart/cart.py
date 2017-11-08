#!/usr/bin/env python3
# -*- utf-8 -*-

import numpy as np


class TreeNode:
    def __init__(self, feat, val, right, left):
        self.feat2split = feat
        self.value_split = val
        self.right_branch = right
        self.left_branch = left


def load_data(file_name):
    data_mat = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line_arr = line.strip().split('\t')
            line_arr = list(map(float, line_arr))
            data_mat.append(line_arr)
    return data_mat


def bin_split_data(data_set, feat_dim, value):
    mat_left = data_set[np.nonzero(data_set[:, feat_dim] > value)[0], :]
    mat_right = data_set[np.nonzero(data_set[:, feat_dim] <= value)[0], :]
    return mat_left, mat_right


def reg_leaf(data_set):
    # 回归叶节点
    return np.mean(data_set[:, -1])


def reg_err(data_set):
    # 回归误差
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    feat_dim, val = choose_best_split(data_set, leaf_type, err_type, ops)
    if feat_dim is None:
        return val
    ret_tree = {'split_dim': feat_dim, 'split_val': val}
    l_set, r_set = bin_split_data(data_set, feat_dim, val)
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    tol_s = ops[0]
    tol_n = ops[1]
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    m, n = np.shape(data_set)
    s = err_type(data_set)
    best_s = np.inf
    best_dim = 0
    best_val = 0
    for feat_dim in range(n - 1):
        for split_val in set(data_set[:, feat_dim].T.A.tolist()[0]):
            mat_left, mat_right = bin_split_data(data_set, feat_dim, split_val)
            if np.shape(mat_left)[0] < tol_n or np.shape(mat_right)[0] < tol_n:
                continue
            new_s = err_type(mat_left) + err_type(mat_right)
            if new_s < best_s:
                best_dim = feat_dim
                best_val = split_val
                best_s = new_s
    if (s - best_s) < tol_s:
        return None, leaf_type(data_set)
    mat_left, mat_right = bin_split_data(data_set, best_dim, best_val)
    if np.shape(mat_left)[0] < tol_n or np.shape(mat_right)[0] < tol_n:
        return None, leaf_type(data_set)
    return best_dim, best_val


if __name__ == '__main__':
    pass
