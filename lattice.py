#!/usr/bin/env python3
from collections import defaultdict

__author__ = 'dvorkjoker,jodaiber'
"""
Script to train weights for features for decomposition lattice.
"""

import ast

def edge_to_split(edge):
    return edge[0], edge[1]

class Lattice(object):
    def __init__(self, arg):

        self.arcs_to = defaultdict(list)
        self.arc_offsets = {}

        # parse from string, if needed
        if isinstance(arg, str):
            arg = ast.literal_eval(arg)

        self.lattice = arg
        self.features = defaultdict(lambda: (100, 0.05))
        self.edges = set()
        for (key, edges) in self.lattice.items():
            if edges:
                for edge in edges:
                    (edge_from, edge_to, prefix, rank, similarity) = edge
                    self.arcs_to[edge_to].append(edge)
                    self.edges.add(edge)
                    split = edge_to_split(edge)
                    if split not in self.features or self.features[split][0] > rank:
                        self.features[split] = (rank, similarity)


    def get_splits(self):  # [(0,3), ...]
        return sorted(set(map(edge_to_split, self.edges)))

    def splits_from(self, i):
        if i not in self.lattice:
            return []
        else:
            return map(edge_to_split, self.lattice[i])

    def splits_to(self, i):
        if i == 0:
            return [(0,0)]
        else:
            return map(edge_to_split, self.arcs_to[i])

    def get_features(self, split, compound):
        return self.features[split]

