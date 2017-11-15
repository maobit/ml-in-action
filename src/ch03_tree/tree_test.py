#!/usr/bin/env python

import trees as trees
import tree_plotter as tree_plotter


def parse_data(filename):
    with open(filename) as f:
        lenses = [x.strip().split('\t') for x in f.readlines()]
    lense_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses, lense_labels


if __name__ == '__main__':
    lenses, lense_labels = parse_data('../../data/ch3/lenses.txt')
    lenses_tree = trees.create_tree(lenses, lense_labels)
    tree_plotter.create_plot(lenses_tree)
