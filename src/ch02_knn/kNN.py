#!/usr/bin/env python

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
import os


def createDataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 在行方向上重复 dataSetSize 次
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    # 读取数据，转化为分类器使用的数据结构
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOflines = len(arrayOfLines)
    returnMat = np.zeros((numberOflines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def plot_data(mat, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(mat[:, 1], mat[:, 2], 15.0 * np.array(labels), 15.0 * np.array(labels))
    plt.show()


def auto_norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_dataset = dataset - np.tile(min_vals, (m, 1))
    norm_dataset = norm_dataset / np.tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals


def dating_class_test():
    test_ratio = 0.1
    dating_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_mat)
    m = norm_mat.shape[0]
    num_test = int(m * test_ratio)
    err_cnt = 0
    for i in range(num_test):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test:, :], dating_labels[num_test:], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            err_cnt += 1
    print('the total error rate is: %f' % (err_cnt / float(num_test)))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('percentage of time spent playing video games? '))
    ff_miles = float(input('frequent filer miles earned per year? '))
    ice_cream = float(input('liters of ice cream consumed per year? '))
    dating_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_mat)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print('You will probbaly like this person: ', result_list[classifier_result - 1])


def img2vector(filename):
    return_vec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vec[0, 32 * i + j] = int(line_str[i])
    return return_vec


def digits_classify():
    digits_labels = []
    train_files = os.listdir('../../data/digits/trainingDigits')
    m = len(train_files)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name = train_files[i]
        file_str = file_name.split('.')[0]
        class_num = int(file_str.split('_')[0])
        digits_labels.append(class_num)
        training_mat[i, :] = img2vector('../../data/digits/trainingDigits/%s' % file_name)
    test_files = os.listdir('../../data/digits/testDigits')
    err_cnt = 0
    m_test = len(test_files)
    for i in range(m_test):
        file_name = test_files[i]
        file_str = file_name.split('.')[0]
        class_num = int(file_str.split('_')[0])
        vec_under_test = img2vector('../../data/digits/testDigits/%s' % file_name)
        classifier_result = classify0(vec_under_test, training_mat, digits_labels, 3    )
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, class_num))
        if classifier_result != class_num:
            err_cnt += 1
    print('the total number of errors is: %d' % err_cnt)
    print('the total error rate is: %f' % (err_cnt / float(m_test)))


if __name__ == '__main__':
    # classify_person()
    digits_classify()
