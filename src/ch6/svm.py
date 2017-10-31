#!/usr/bin/env python3
# -*- utf-8 -*-

import numpy as np


def load_data(file_name):
    data_mat = []
    label_mat = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line_arr = line.strip().split('\t')
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[-1]))
    return data_mat, label_mat


def select_j_rand(i, m):
    # 随机选取j
    j = i
    while j == i:  # 直到j和i不同
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(a_j, h, l):
    # 对于超出范围的aj进行截断
    if a_j > h:
        a_j = h
    if a_j < l:
        a_j = l
    return a_j


def smo_simple(data_mat, class_mat, c, toler, max_iter):
    data_matrix = np.mat(data_mat)
    label_mat = np.mat(class_mat).transpose()
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.zeros((m, 1))
    iteration = 0
    while iteration < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            f_xi = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            e_i = f_xi - float(label_mat[i])
            if (label_mat[i] * e_i < -toler and alphas[i] < c) or (label_mat[i] * e_i > toler and alphas[i] > 0):
                j = select_j_rand(i, m)
                f_xj = float(np.multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                e_j = f_xj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    l = max(0, alphas[i] - alphas[j])
                    h = min(c, c + alphas[i] - alphas[j])
                else:
                    l = max(0, alphas[i] + alphas[j] - c)
                    h = min(c, alphas[i] + alphas[j])
                if l == h:
                    print('l == h')
                    continue
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T - \
                      data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] = alphas[j] - label_mat[j] * (e_i - e_j) / eta
                alphas[j] = clip_alpha(alphas[j], h, l)
                if abs(alphas[j] - alpha_j_old < 0.00001):
                    print('j not moving enough')
                    continue
                alphas[i] = alphas[i] + label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - e_i - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - e_j - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print('iteration: %d i:%d, pairs changed %d' % (iteration, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iteration += 1
        else:
            iteration = 0
        print('iteration number: %d' % iteration)
    return b, alphas


if __name__ == '__main__':
    pass
