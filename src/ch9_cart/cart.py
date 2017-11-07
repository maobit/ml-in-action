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
            line_arr = map(float, line_arr)
            data_mat.append(line_arr)
    return data_mat


def bin_split_data(data_set, feat_dim, value):
    mat_right = data_set[np.nonzero(data_set[:, feat_dim] > value)[0], :][0]
    mat_left = data_set[np.nonzero(data_set[:, feat_dim] <= value)[0], :][0]
    return mat_left, mat_right




if __name__ == '__main__':
    pass
