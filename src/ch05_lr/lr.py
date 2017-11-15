#!/usr/bin/env python3
# -*- utf-8 -*-

import numpy as np


def load_data():
    data_mat = []
    label_mat = []
    fr = open('../../data/ch5/testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return np.array(data_mat), np.array(label_mat)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def grad_ascent(data_mat_in, class_labels):
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)  # 书中并没有具体讲解如何得到梯度更新的公式
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i] == 1):
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = np.reshape((-weights[0] - weights[1] * x) / weights[2], (np.shape(x)[0], 1))
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stoc_grad_ascent0(data_matrix, class_labels):
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]
    return weights


def stoc_grad_ascent1(data_matrix, class_labels, number_iter=150):
    m, n = np.shape(data_matrix)
    weights = np.ones(n)
    for j in range(number_iter):
        data_idx = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_idx = int(np.random.uniform(0, len(data_idx)))
            h = sigmoid(np.sum(data_matrix[rand_idx] * weights))
            error = class_labels[rand_idx] - h
            weights = weights + alpha * error * data_matrix[rand_idx]
            del (data_idx[rand_idx])
    return weights


def classify(x, weights):
    prob = sigmoid(np.sum(x * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colic_test():
    fr_train = open('../../data/ch5/horseColicTraining.txt')
    fr_test = open('../../data/ch5/horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        cur_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(cur_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(cur_line[-1]))
    train_weights = stoc_grad_ascent1(np.array(training_set), training_labels, 500)
    err_cnt = 0
    num_test = 0
    for line in fr_test.readlines():
        num_test += 1
        cur_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(cur_line[i]))
        if int(classify(np.array(line_arr), train_weights) != float(cur_line[-1])):
            err_cnt += 1
    err_rate = (float(err_cnt) / num_test)
    print('the error rate of this test is: %f' % err_rate)
    return err_rate


def multi_test():
    num_tests = 10
    err_sum = 0.0
    for k in range(num_tests):
        err_sum += colic_test()
    print('after %d iterations the average error rate is: %f' % (num_tests, err_sum / float(num_tests)))
