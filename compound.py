from lattice import *

class Compound:

    def __init__(self, string, gold_splits, predicted_lattice):
        self.gold_splits = gold_splits
        self.predicted_lattice = predicted_lattice
        self.string = string

    def get_gold_splits(self):
        return self.gold_splits

