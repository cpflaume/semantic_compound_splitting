from viterbi import split_to_footprint

class Compound:

    def __init__(self, string, gold_lattice, predicted_lattice):
        self.gold_lattice = gold_lattice
        self.predicted_lattice = predicted_lattice
        self.string = string

    def get_splits(self):
        return [split_to_footprint(self.string, s) for s in self.gold_lattice.get_splits()]

    def get_viterbi_splits(self, weights):
        return [split_to_footprint(self.string, s) for s in self.predicted_lattice.get_viterbi(weights)]