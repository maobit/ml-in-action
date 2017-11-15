#!/usr/bin/env python3
# -*- utf-8 -*-

import numpy as np


def load_simple_data():
    data_mat = np.matrix([[1., 2.1],
                          [2., 1.1],
                          [1.3, 1.],
                          [1., 1.],
                          [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def stump_classify(data_mat, dimen, thresh_val, thresh_ineq):
    ret_arr = np.ones((np.shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        ret_arr[data_mat[:, dimen] <= thresh_val] = -1.0
    else:
        ret_arr[data_mat[:, dimen] > thresh_val] = 1.0
    return ret_arr


def build_stump(data_arr, class_labels, d):
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_mat)
    num_steps = 10
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):
        range_min = data_mat[:, ].min()
        range_max = data_mat[:, ].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_err = d.T * err_arr
                print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted erroris %.3f' % (
                i, thresh_val, inequal, weighted_err))
                if weighted_err < min_error:
                    min_error = weighted_err
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est


def adaboost_train_ds(data_arr, class_labels, num_iter=40):
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    d = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_iter):
        best_stump, error, class_est = build_stump(data_arr, class_labels, d)
        print('d:', d.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        print('class_est: ', class_est.T)
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
        d = np.multiply(d, np.exp(expon))
        d = d / d.sum()
        agg_class_est += alpha * class_est
        print('agg_class_est: ', agg_class_est.T)
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        err_rate = agg_errors.sum() / m
        print('total error: ', err_rate)
        if err_rate == 0.0:
            break
    return weak_class_arr, agg_class_est


def ada_classify(data2class, classifier_arr):
    data_mat = np.mat(data2class)
    m = np.shape(data_mat)[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_mat, classifier_arr[i]['dim'], classifier_arr[i]['thresh'], classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha'] * class_est
        print(agg_class_est)
    return np.sign(agg_class_est)


def load_data(file_name):
    num_feat = len(open(file_name).readline().split('\t'))
    data_mat = []
    label_mat = []
    with open(file_name) as fr:
        for line in fr.readlines():
            line_arr = []
            cur_line = line.strip().split('\t')
            for i in range(num_feat - 1):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def plot_roc(pred_strengths, class_labels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    y_sum = 0.0
    num_pos_class = np.sum(np.array(class_labels) == 1.0)
    y_step = 1 / float(num_pos_class)
    x_step = 1 / (len(class_labels) - num_pos_class)
    sorted_indicies = pred_strengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sorted_indicies.tolist()[0]:
        if class_labels[index] == 1.0:
            del_x = 0
            del_y = y_step
        else:
            del_x = x_step
            del_y = 0
            y_sum += cur[1]
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
        cur = (cur[0] - del_x, cur[1] - del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area Under the Curve is:', y_sum * x_step)


if __name__ == '__main__':
    pass
