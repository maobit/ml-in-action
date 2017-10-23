#!/usr/bin/env python

import numpy as np
import os
from k_nearest_neighbor import KNearestNeighbor


def img2vector(filename):
    return_vec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vec[0, 32 * i + j] = int(line_str[i])
    return return_vec


def digits_classify(classifier):
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
    # 使用训练集进行训练
    classifier.fit(training_mat, digits_labels)

    test_files = os.listdir('../../data/digits/testDigits')
    err_cnt = 0
    m_test = len(test_files)
    for i in range(m_test):
        file_name = test_files[i]
        file_str = file_name.split('.')[0]
        class_num = int(file_str.split('_')[0])
        vec_under_test = img2vector('../../data/digits/testDigits/%s' % file_name)
        classifier_result = classifier.predict(vec_under_test)
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, class_num))
        if classifier_result != class_num:
            err_cnt += 1
    print('the total number of errors is: %d' % err_cnt)
    print('the total error rate is: %f' % (err_cnt / float(m_test)))

if __name__ == '__main__':
    knn = KNearestNeighbor(k=10)
    digits_classify(classifier=knn)
