from lattice import *

class Compound:

    def __init__(self, string, gold_lattice, predicted_lattice):
        self.gold_lattice = gold_lattice
        self.predicted_lattice = predicted_lattice
        self.string = string

    def get_gold_splits(self):
        return split_to_footprint(self.string, self.gold_lattice.get_splits()[0])

    def get_viterbi_splits(self, weights):
        return split_to_footprint(self.string, all_splits(self.predicted_lattice.get_viterbi(weights))[0])