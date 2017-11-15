#!/usr/bin/env python3
# -*- utf-8 -*-

import numpy as np


def load_data(file_name, delim='\t'):
    with open(file_name) as fr:
        string_arr = [line.strip().split(delim) for line in fr.readlines()]
    return np.mat([list(map(float, line)) for line in string_arr])


def pca(data_mat, top_n_feat=9999999):
    means = np.mean(data_mat, axis=0)
    mean_removed = data_mat - means
    cov_mat = np.cov(mean_removed, rowvar=0)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    eig_val_ind = np.argsort(eig_vals)
    eig_val_ind = eig_val_ind[: -(top_n_feat + 1): -1]
    red_eig_vects = eig_vects[:, eig_val_ind]
    low_dim_data_mat = mean_removed * red_eig_vects
    recon_mat = (low_dim_data_mat * red_eig_vects.T) + means
    return low_dim_data_mat, recon_mat

if __name__ == '__main__':
    pass
