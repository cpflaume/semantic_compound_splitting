from __future__ import division
import codecs
import numpy as np
import sys
import matplotlib.pyplot as plt
from lattice import Lattice
from compound import Compound
from itertools import chain
import ast
from collections import defaultdict
from viterbi_decompounder import ViterbiDecompounder


def get_prev_split(splits, split):
    i = splits.index(split)
    if i == 0:
        return (0, 0)
    else:
        return splits[i - 1]


class StructuredPerceptron:
    def __init__(self, epochs=10, eta=.0001):

        self.decoder = ViterbiDecompounder()
        self.parameters_for_epoch = []

        self.n_epochs = epochs
        self.eta = eta

        self.n_features = ViterbiDecompounder.n_features

    def train(self, data, heldout, verbose=0, run_label=None):

        self.decoder.w = np.ones(self.n_features, dtype=float) / self.n_features
        print >> sys.stderr, "Start weights: %s" % self.decoder.w

        training_accuracy = [0.0]
        heldout_accuracy = [0.0]

        for i_epoch in xrange(self.n_epochs):

            tp, fp, fn = 0, 0, 0

            for compound in data:
                tp, fp, fn = self.train_one(compound, tp, fp, fn)

            self.parameters_for_epoch.append(self.decoder.w.copy())

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * ((precision * recall) / (precision + recall))
            training_accuracy.append(f1)

            if verbose == 1:
                acc = self.test(heldout)
                heldout_accuracy.append(acc)

            print "Training", training_accuracy
            # Stop if the error on the training data does not decrease
            if training_accuracy[-1] <= training_accuracy[-2]:
                break

            print >> sys.stderr, "Weights: %s" % self.decoder.w
            print >> sys.stderr, "Epoch %i, F1: %f" % (i_epoch, f1)

        # Average!
        averaged_parameters = 0
        for epoch_parameters in self.parameters_for_epoch:
            averaged_parameters += epoch_parameters
        averaged_parameters /= len(self.parameters_for_epoch)

        self.decoder.w = averaged_parameters

        # Finished training
        self.trained = True

        if verbose == 1:
            print "Heldout accs:", str(heldout_accuracy)
            print self.decoder.w

        # Export training info in verbose mode:
        if verbose == 2:
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

    def train_one(self, compound, tp, fp, fn):

        # Returns a list of tuples with (start, stop) position
        predicted_splits = self.decoder.viterbi_decode(compound)

        gold_splits = compound.get_gold_splits()
        gold_splits_set = set(gold_splits)
        predicted_splits_set = set(predicted_splits)

        for split in gold_splits_set.union(predicted_splits_set):
            if split in predicted_splits_set and split in gold_splits_set:  # Do nothing
                tp += 1

            if split[1] == len(compound.string) and split[0] != 0:  # Ignore the final artificial path
                continue

            if split in predicted_splits_set and split not in gold_splits_set:  # This is a bad split!
                prev_split = get_prev_split(predicted_splits, split)

                predicted_split_features = self.decoder.fs(compound, prev_split, split, compound.predicted_lattice)
                print >> sys.stderr, "Pred fs:", predicted_split_features
                self.decoder.w -= self.eta * (self.decoder.w * predicted_split_features)

                fp += 1

            if split not in predicted_splits_set and split in gold_splits_set:  # This split should have been there!
                prev_split = get_prev_split(gold_splits, split)

                gold_split_features = self.decoder.fs(compound, prev_split, split, compound.predicted_lattice)
                print >> sys.stderr, "Gold fs:", gold_split_features
                print >> sys.stderr, "w:", self.decoder.w
                self.decoder.w += self.eta * (self.decoder.w * gold_split_features)

                fn += 1

        return tp, fp, fn

    def test(self, compounds):
        tp, fp, fn = 0, 0, 0

        for compound in compounds:
            z = self.decoder.viterbi_decode(compound)

            gold_splits = set(compound.gold_splits)
            for split in z:
                if split in gold_splits:
                    tp += 1
                else:
                    fp += 1

            for gold_split in gold_splits:
                if gold_split not in z:
                    fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ((precision * recall) / (precision + recall))

        print "Test Precision: %f" % recall
        print "Test Recall: %f" % precision
        print "Test F1: %f" % f1

        return f1


def split_gold(l):
    l1 = l.split("|||")[0].strip()
    l2 = l.split("|||")[1].strip()[2:].split(" ")

    ps = []
    for lp in l2:
        assert l1.count(lp) == 1
        ps.append(l1.find(lp))

    splits = []
    for i in range(len(ps)):
        if i == 0:
            pass
        elif i == 1:
            splits.append((0, ps[i]))
        else:
            splits.append((ps[i - 1], ps[i]))

        if i == len(ps) - 1:
            splits.append((ps[i], len(l1)))

    return sorted(splits)


def all_gold_splits_in_lattice(c):
    lattice_splits = set(c.predicted_lattice.get_splits())
    return all([gsplit in lattice_splits for gsplit in c.get_gold_splits()])


def correct_in_lattice(cs):
    split_in_lattice = [all_gold_splits_in_lattice(c) for c in cs]
    return sum(split_in_lattice) / len(split_in_lattice)


HELDOUT_SIZE = 75
if __name__ == '__main__':
    compound_names = codecs.open("data/cdec_nouns", encoding="utf-8").readlines()
    compounds_gold = codecs.open("data/cdec_nouns.references", encoding="utf-8").readlines()
    compounds_pred = codecs.open("data/cdec_nouns.lattices", encoding="utf-8").readlines()

    compounds = map(lambda (compound, lineGold, latticePredicted):
                    Compound(compound.strip(), split_gold(lineGold), Lattice(latticePredicted)),
                    zip(compound_names, compounds_gold, compounds_pred))

    import random

    random.shuffle(compounds)

    train, heldout = compounds[HELDOUT_SIZE:], compounds[:HELDOUT_SIZE]

    for c in train:
        if not all_gold_splits_in_lattice(c):
            print "  Unreachable training instance:", c.string, "Gold:", c.gold_splits, "Predicted:", c.predicted_lattice.get_splits()

    trainer = StructuredPerceptron(epochs=10)
    trainer.train(train, heldout, verbose=1)
    print "% Gold path in the lattice: ", correct_in_lattice(heldout)

    import yaml

    with open('weights', 'w') as outfile:
        outfile.write(yaml.dump(trainer.decoder.w.tolist(), default_flow_style=True))
