from __future__ import division
import codecs
import numpy as np
import sys
import matplotlib.pyplot as plt
from lattice import Lattice, split_to_footprint
from compound import Compound
from itertools import chain
import ast


def ends_at(split):
    return split[1]


def get_prev_split(splits, split):
    i = splits.index(split)
    if i == 0:
        return (0, 0)
    else:
        return splits[i - 1]


class StructuredPerceptron:
    def __init__(self, epochs=10, eta=1.):

        self.parameters_for_epoch = []

        self.n_epochs = epochs
        self.eta = eta

        self.n_features = 7

    def train(self, data, heldout, verbose=False, run_label=None):

        self.w = np.ones(self.n_features, dtype=float) / self.n_features

        training_accuracy = [0.0]
        heldout_accuracy = [0.0]

        for i_epoch in xrange(self.n_epochs):

            tp, fp, fn = 0, 0, 0

            for compound in data:
                tp, fp, fn = self.train_one(compound, tp, fp, fn)

            self.parameters_for_epoch.append(self.w.copy())

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * ((precision * recall) / (precision + recall))
            training_accuracy.append(f1)

            if verbose:
                _, _, acc = self.test(heldout)
                heldout_accuracy.append(acc)

            # Stop if the error on the training data does not decrease
            if training_accuracy[-1] <= training_accuracy[-2]:
                break

            print >> sys.stderr, "Epoch %i, Accuracy: %f" % (i_epoch, f1)

        # Average!
        averaged_parameters = 0
        for epoch_parameters in self.parameters_for_epoch:
            averaged_parameters += epoch_parameters
        averaged_parameters /= len(self.parameters_for_epoch)

        self.w = averaged_parameters

        # Finished training
        self.trained = True

        # Export training info in verbose mode:
        if verbose:
            x = np.arange(0, len(training_accuracy), 1.0)
            plt.plot(x, training_accuracy, marker='o', linestyle='--', color='r', label='Training')
            plt.plot(x, heldout_accuracy, marker='o', linestyle='--', color='b', label='Heldout')

            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Heldout Accuracy')

            plt.ylim([0.9, 1.0])

            plt.legend(bbox_to_anchor=(1., 0.2))

            plt.savefig('eval/%s_training.png' % run_label)

            plt.close()

    def arc_score(self, compound, prev_split, split, lattice):
        return np.dot(self.w, self.fs(compound, prev_split, split, lattice))

    def viterbi_decode(self, compound):
        alphas = np.zeros(len(compound.string), dtype=float)
        path = {}

        START_SPLIT = (0, 0)
        END_SPLIT = (len(compound.string),len(compound.string))

        lattice = compound.predicted_lattice

        for i in xrange(0, len(compound.string)):
            new_path = {}

            for split in lattice.splits_from(i):
                (alphas[i], b) = max(
                    [(alphas[ends_at(split_last)] + self.arc_score(compound, split_last, split, lattice), split_last)
                     for split_last in lattice.splits_to(i)])
                new_path[split] = path[b] + [split]

            path = new_path

        return path[END_SPLIT]

    def fs(self, compound, prev_split, split, lattice):
        # Base features on the lattice:
        # (0, 1.0, 0, 1) rank, cosine, split penalty, is_no_split

        # Additional features:

        base_features = list(lattice.get_features(split, compound))
        base_features.append(split[1] - split[0])  # Length of the split
        base_features.append(prev_split[1] - prev_split[0])  # Length of the previous split
        base_features.append(1.0)  # Bias

        return np.array(base_features)

    def train_one(self, compound, tp, fp, fn):

        # Returns a list of tuples with (start, stop) position
        predicted_splits = self.viterbi_decode(compound)

        gold_splits = compound.get_gold_splits()
        gold_splits_set = set(gold_splits)
        predicted_splits_set = set(predicted_splits)

        for split in gold_splits_set.union(predicted_splits_set):
            if split in predicted_splits_set and split in gold_splits_set:  # Do nothing
                tp += 1

            if split in predicted_splits_set and split not in gold_splits_set:  # This is a bad split!
                prev_split = get_prev_split(predicted_splits, split)

                predicted_split_features = self.fs(compound, prev_split, split, compound.predicted_lattice)
                self.w -= self.eta * (self.w * predicted_split_features)

                fp += 1

            if split not in predicted_splits_set and split in gold_splits_set:  # This split should have been there!
                prev_split = get_prev_split(gold_splits, split)

                gold_split_features = self.fs(compound, prev_split, split, compound.gold_lattice)
                self.w += self.eta * (self.w * gold_split_features)

                self.w[gold_split_features] += self.eta

                fn += 1

        return tp, fp, fn

    def test(self, compounds):
        tp, fp, fn = 0, 0, 0

        for compound in compounds:
            z = self.viterbi_decode(compound)

            for split in z:
                if compound.gold_lattice.contains(split):
                    tp += 1
                else:
                    fp += 1

            for gold_split in compound.gold_lattice.gold_path:
                if gold_split not in z:
                    fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))

        print "Test Precision: %f\n" % recall
        print "Test Recall: %f\n" % precision
        print "Test F1: %f\n" % f1

        return f1


HELDOUT_SIZE = 50

if __name__ == '__main__':
    compound_names = codecs.open("data/cdec_nouns").readlines()
    compounds_gold = codecs.open("data/cdec_nouns.ref.lattices").readlines()
    compounds_pred = codecs.open("data/cdec_nouns.lattices").readlines()

    compounds = map(
        lambda (compound, latticeGold, latticePredicted): Compound(compound, Lattice(latticeGold),
                                                                   Lattice(latticePredicted)),
        zip(compound_names, compounds_gold, compounds_pred)
    )

    train, heldout = compounds[HELDOUT_SIZE:], compounds[:HELDOUT_SIZE]

    trainer = StructuredPerceptron(epochs=10)
    trainer.train(train, heldout, verbose=True)
