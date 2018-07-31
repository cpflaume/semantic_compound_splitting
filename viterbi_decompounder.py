
import numpy as np
import yaml
from collections import defaultdict


def ends_at(split):
    return split[1]


def starts_at(split):
    return split[0]


class ViterbiDecompounder:
    def __init__(self):
        pass

    def load_weights(self, weights):
        self.w = np.array(weights)

    def viterbi_decode(self, compound):

        # print "\n"*4
        # print compound.string

        alphas = [defaultdict(lambda: 0.0) for _ in range(len(compound.string) + 1)]
        path = {(0, 0): []}
        alphas[0] = defaultdict(lambda: 1.0)

        lattice = compound.predicted_lattice

        for split in lattice.splits_from(0):
            path[split] = [(0, 0)]
            # print "From 0:", str(split)

        for i in range(len(compound.string) + 1):
            new_path = path

            for split in lattice.splits_from(i):
                if len(lattice.splits_to(i)) > 0 and len(lattice.splits_from(i)) > 0:
                    (alphas[split[1]][split], b) = max([(alphas[ends_at(split_last)][split_last] + self.arc_score(
                        compound, split_last, split, lattice), split_last) for split_last in lattice.splits_to(i)])

                    new_path[split] = path[b] + [split]

            path = new_path

        # print "path",  path

        # print "Final alphas"
        # for i, a in enumerate(alphas):
        #    print i, a

        f = len(compound.string)
        (_, b) = max([(alphas[split_last[1]][split_last], split_last) for split_last in lattice.splits_to(f)])

        return path[b]

    def arc_score(self, compound, prev_split, split, lattice):
        return np.dot(self.w, self.fs(compound, prev_split, split, lattice))

    def fs(self, compound, prev_split, split, lattice):
        # Base features on the lattice:
        # (0, 1.0, 0, 1) rank, cosine, split penalty, is_no_split

        # Additional features:

        segment = compound.string[split[0]:split[1]]
        is_default = split[1] - split[0] == len(compound.string)

        base_features = list(lattice.get_features(split, compound))

        if is_default:
            base_features[0] = 0
            base_features[1] = 1.0

        base_features[0] = 1.0 - (base_features[0] / 100)

        base_features.append(1 if is_default else 0)
        base_features.append(0 if is_default else 1)

        base_features.append(1 if (split[1] - split[0]) > 12 and not is_default else 0)  # Length of the split
        base_features.append(1 if (split[1] - split[0]) < 4 else 0)  # Length of the split

        # base_features.append(1 if segment.endswith("es") or segment.endswith("s") else 0)

        base_features.append((split[1] - split[0]) / len(compound.string))  # Length of the split

        # base_features.append(prev_split[1] - prev_split[0])  # Length of the previous split
        base_features.append(1.0)  # Bias

        return np.array([0 if f == 0 else 1 for f in base_features])

    n_features = 8
