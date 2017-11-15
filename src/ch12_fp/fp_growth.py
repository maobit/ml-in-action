#!/usr/bin/env python3
# -*- utf-8 -*-

import numpy as np


class TreeNode:
    def __init__(self, name, num_occur, parent):
        self.name = name
        self.count = num_occur
        self.node_link = None
        self.parent = parent
        self.children = {}

    def inc(self, num_occur):
        self.count += num_occur

    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def create_tree(data_set, min_sup=1):
    header = {}
    for trans in data_set:
        for item in trans:
            header[item] = header.get(item, 0) + data_set[trans]
    header = {k: v for k, v in header.items() if v >= min_sup}
    freq_item_set = set(header.keys())
    if len(freq_item_set) == 0:
        return None, None
    for k in header:
        header[k] = [header[k], None]
    ret_tree = TreeNode('Null Set', 1, None)
    for trans, count in data_set.items():
        local_d = {}
        for item in trans:
            if item in freq_item_set:
                local_d[item] = header[item][0]
        if len(local_d) > 0:
            ordered_items = [v[0] for v in sorted(local_d.items(), key=lambda p: p[1], reverse=True)]
            update_tree(ordered_items, ret_tree, header, count)
    print('222')
    return ret_tree, header


def update_tree(items, in_tree, header, count):
    if items[0] in in_tree.children:
        in_tree.children[items[0]].inc(count)
    else:
        in_tree.children[items[0]] = TreeNode(items[0], count, in_tree)
    if header[items[0]][1] is None:
        header[items[0]][1] = in_tree.children[items[0]]
    else:
        update_header(header[items[0]][1], in_tree.children[items[0]])
    if len(items) > 1:
        update_tree(items[1::], in_tree.children[items[0]], header, count)


def update_header(node2test, target_node):
    while node2test.node_link is not None:
        node2test = node2test.node_link
    node2test.node_link = target_node


def load_simple_data():
    return [['r', 'z', 'h', 'j', 'p'],
            ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
            ['z'],
            ['r', 'x', 'n', 'o', 's'],
            ['y', 'r', 'x', 'z', 'q', 't', 'p'],
            ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]


def create_init_set(data_set):
    ret_dict = {}
    for trans in data_set:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


if __name__ == '__main__':
    pass
