#!/usr/bin/env python

import operator


class KNearestNeighbor:
    def __init__(self, k):
        self.k = k
        self.training_data = None
        self.labels = None

    def fit(self, training_data, labels):
        self.training_data = training_data
        self.labels = labels

    def predict(self, in_x):
        label_cnt = {}
        distances = self._l2_distance(self.training_data, in_x)
        sorted_indices = distances.argsort()
        for i in range(self.k):
            voted_label = self.labels[sorted_indices[i]]
            label_cnt[voted_label] = label_cnt.get(voted_label, 0) + 1
        sorted_label_cnt = sorted(label_cnt.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_label_cnt[0][0]

    @staticmethod
    def _l2_distance(x1, x2):
        distances = (x1 - x2) ** 2  # 使用numpy的广播机制
        distances = distances.sum(axis=1)
        distances = distances ** 0.5
        return distances
