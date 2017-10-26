#!/usr/bin/env python

import numpy as np


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


def bag_of_words2vec(vocab_list, input_set):
    # 词袋模型
    ret_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            ret_vec[vocab_list.index(word)] += 1
    return ret_vec


def text_parse(big_string):
    import re
    list_of_tokens = re.split(r'\W*', big_string)
    return [token.lower() for token in list_of_tokens if len(token) > 2]


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        word_list = text_parse(open('../../data/ch4/email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open('../../data/ch4/email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    train_set = range(50)
    test_set = []
    for i in range(10):
        rand_index = int(np.random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_index])
        del(train_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in train_set:
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb0(np.array(train_mat), np.array(train_classes))
    error_cnt = 0
    for doc_index in test_set:
        word_vec = set_of_words2vec(vocab_list, doc_list[doc_index])
        if classify_nb(word_vec, p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_cnt += 1
    print('the error rate is: %f' % (float(error_cnt) / len(test_set)))

