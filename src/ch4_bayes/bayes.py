#!/usr/bin/env python

import numpy as np
import operator


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    # 创建词表
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    ret_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] = 1
        else:
            print('the word "%s"  is not in my vocabulary' % word)
    return ret_vec


def train_nb0(train_mat, train_category):
    # 计算每一个单词出现的在给定类别的条件下的概率
    num_train_docs = len(train_mat)
    num_words = len(train_mat[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words)  # 为防止计算的后验概率为 0
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 0:
            p0_num += train_mat[i]
            p0_denom += sum(train_mat[i])
        else:
            p1_num += train_mat[i]
            p1_denom += sum(train_mat[i])
    p1_vect = np.log(p1_num / p1_denom)
    p0_vect = np.log(p0_num / p0_denom)

    return p0_vect, p1_vect, p_abusive


def classify_nb(vec_to_classify, p0_vec, p1_vec, p_class1):
    # 针对二分类问题，如果是多分类需要进行适当的修改
    p1 = sum(vec_to_classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec_to_classify * p0_vec) + np.log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def test_nb():
    list_posts, list_classes = load_data_set()
    vocab_list = create_vocab_list(list_posts)
    train_mat = []
    for post in list_posts:
        train_mat.append(set_of_words2vec(vocab_list, post))
    p0_v, p1_v, p_class1 = train_nb0(train_mat, list_classes)
    test_entry = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_words2vec(vocab_list, test_entry))
    print('%s classified as %d' % (str(test_entry), classify_nb(this_doc, p0_v, p1_v, p_class1)))
    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_words2vec(vocab_list, test_entry))
    print('%s classified as %d' % (str(test_entry), classify_nb(this_doc, p0_v, p1_v, p_class1)))
