#!/usr/bin/env python3
# -*- utf-8 -*-

import numpy as np


def load_data(file_name):
    data_mat = []
    with open(file_name) as fr:
        for line in fr.readlines():
            cur_line = line.strip().split('\t')
            data_mat.append(list(map(float, cur_line)))
    return data_mat


def dist_eclud(vec_a, vec_b):
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


def rand_centroid(data_set, k):
    n = np.shape(data_set)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = np.min(data_set[:, j])
        range_j = float(np.max(data_set[:, j]) - min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids


def k_means(data_set, k, dist_meas=dist_eclud, create_centroid=rand_centroid):
    m = np.shape(data_set)[0]
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroids = create_centroid(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_idx = -1
            for j in range(k):
                dist_ji = dist_meas(centroids[j, :], data_set[i, :])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_idx = j
            if cluster_assment[i, 0] != min_idx:
                cluster_changed = True
            cluster_assment[i, :] = min_idx, min_dist ** 2
        print(centroids)
        for cent in range(k):
            pts_in_cluster = data_set[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(pts_in_cluster, axis=0)
    return centroids, cluster_assment


def bi_k_means(data_set, k, dist_meas=dist_eclud):
    m = np.shape(data_set)[0]
    cluster_assment = np.mat(np.zeros((m, 2)))
    centroid_init = np.mean(data_set, axis=0).tolist()[0]
    cent_list = [centroid_init]
    for j in range(m):
        cluster_assment[j, 1] = dist_meas(np.mat(centroid_init), data_set[j, :]) ** 2
    while len(cent_list) < k:
        lowest_sse = np.inf
        for i in range(len(cent_list)):
            pts_in_curcluster = data_set[np.nonzero(cluster_assment[:, 0].A == i)[0], :]
            centroid_mat, split_clust_ass = k_means(pts_in_curcluster, 2, dist_meas)
            sse_split = np.sum(split_clust_ass[:, 1])
            sse_not_split = np.sum(cluster_assment[np.nonzero(cluster_assment[:, 0].A != i)[0], 1])
            print('sse_split, and not split', sse_split, sse_not_split)
            if sse_split + sse_not_split < lowest_sse:
                best_cent2split = i
                best_new_cents = centroid_mat
                best_clust_ass = split_clust_ass.copy()
                lowest_sse = sse_split + sse_not_split
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent2split
        print('the bestCentToSplit is:', best_cent2split)
        print('the len of bestClustAss is:', len(best_clust_ass))
        cent_list[best_cent2split] = best_new_cents[0, :]
        cent_list.append(best_new_cents[1, :])
        cluster_assment[np.nonzero(cluster_assment[:, 0].A == best_cent2split)[0], :] = best_clust_ass
    return cent_list, cluster_assment


if __name__ == '__main__':
    pass
