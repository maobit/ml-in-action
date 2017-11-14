#!/usr/bin/env python3
# -*- utf-8 -*-


import numpy as np


def load_data():
    return [[1, 3, 4],
            [2, 3, 5],
            [1, 2, 3, 5],
            [2, 5]]


def create_c1(data_set):
    # 创建C1，即大小为1的所有候选项的集合
    c1 = []
    for trans in data_set:
        for item in trans:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    return list(map(frozenset, c1))


def scan_data(data, ck, min_support):
    ss_cnt = {}
    for tid in data:
        for can in ck:
            if can.issubset(tid):
                if can not in ss_cnt:
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_items = float(len(data))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data


def apriori_gen(lk, k):
    ret_list = []
    len_lk = len(lk)
    for i in range(len_lk):
        for j in range(i + 1, len_lk):
            l1 = list(lk[i])[:k - 2]
            l2 = list(lk[j])[:k - 2]
            l1.sort()
            l2.sort()
            # 前 k-2 个项相同时，两两合并
            if l1 == l2:
                ret_list.append(lk[i] | lk[j])
    return ret_list


def apriori(data_set, min_support=0.5):
    c1 = create_c1(data_set)
    d = list(map(set, data_set))
    l1, support_data = scan_data(d, c1, min_support)
    l = [l1]
    k = 2
    while len(l[k - 2]) > 0:
        ck = apriori_gen(l[k - 2], k)
        lk, sup_k = scan_data(d, ck, min_support)
        support_data.update(sup_k)
        l.append(lk)
        k += 1
    return l, support_data


def generate_rules(l, support_data, min_conf=0.7):
    big_rule_list = []
    for i in range(1, len(l)):
        for freq_set in l[i]:
            h1 = [frozenset([item]) for item in freq_set]
            if i > 1:
                rules_from_conseq(freq_set, h1, support_data, big_rule_list, min_conf)
            else:
                calc_conf(freq_set, h1, support_data, big_rule_list, min_conf)
    return big_rule_list


def calc_conf(freq_set, h, support_data, big_rules, min_conf=0.7):
    pruned_h = []
    for conseq in h:
        conf = support_data[freq_set] / support_data[freq_set - conseq]
        if conf >= min_conf:
            print(freq_set - conseq, '-->', conseq, 'conf:', conf)
            big_rules.append((freq_set - conseq, conseq, conf))
            pruned_h.append(conseq)
    return pruned_h


def rules_from_conseq(freq_set, h, support_data, big_rules, min_conf=0.7):
    m = len(h[0])
    if len(freq_set) > (m + 1):
        hmp1 = apriori_gen(h, m + 1)
        hmp1 = calc_conf(freq_set, hmp1, support_data, big_rules, min_conf)
        if len(hmp1) > 1:
            rules_from_conseq(freq_set, hmp1, support_data, big_rules, min_conf)


if __name__ == '__main__':
    pass
