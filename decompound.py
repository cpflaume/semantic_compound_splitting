#!/usr/bin/env python
# coding: utf-8

import argparse
import fileinput
from compound import Compound
from lattice import Lattice
import multiprocessing

import pickle
import logging
import pdb
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import WhitespaceTokenizer
import sys
import multiprocessing as mp
import codecs
from annoy import AnnoyIndex
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import yaml

from viterbi_decompounder import ViterbiDecompounder

class BaseDecompounder:

    def __init__(self, model_folder, modelSetup, nAccuracy=250, globalNN=500,
            similarityThreshold=0.0, prototype_file="prototypes.p"):

        # Basic Logging:
        self.logger = logging.getLogger('')
        hdlr = logging.FileHandler('decompound.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.DEBUG)
        ########################################

        self.nAccuracy = nAccuracy
        self.similarityThreshold = similarityThreshold
        self.globalNN = globalNN
        self.FUGENLAUTE = modelSetup["FUGENLAUTE"]

        print >> sys.stderr, "Loading prototypes..."

        self.prototypes = pickle.load(open(model_folder + "/" + prototype_file, 'rb'))

        print >> sys.stderr, "Loading KNN search..."
        self.annoy_tree = AnnoyIndex(500)
        self.annoy_tree.load(model_folder + '/tree.ann')

        print >> sys.stderr, "Loading gensim model..."
        self.model = gensim.models.Word2Vec.load_word2vec_format(args.model_folder + '/w2v.bin', binary=True)

        print >> sys.stderr, "Done."


    def decompound(self, inputCompound, offset=0):

        # 1. See if we can deal with compound
        #
        if len(inputCompound) == 0:
            return []

        self.logger.debug('Looking up word %s in Word2Vec model' % inputCompound)
        if inputCompound not in self.model.vocab:  # We haven't vector representation for compound
            self.logger.debug('ERROR COULDNT FIND KEY %s IN WORD2VEC MODEL' % inputCompound)
            return []
        self.logger.debug('Found key in index dict for word %s' % inputCompound)
        inputCompoundRep = self.model[inputCompound]
        inputCompoundIndex = self.model.vocab[inputCompound].index

        # 2. See if we have prefixes of compound in vocabulary
        #
        self.logger.info('Getting all matching prefixes')
        prefixes = set()
        for prefix in self.prototypes.keys():
            if len(inputCompound) > len(prefix) and inputCompound.startswith(prefix):
                prefixes.add(prefix)
        self.logger.debug('Possible prefixes: %r' % prefixes)
        if len(prefixes) == 0:  # cannot split
            return []

        # 3. Get all possible splits (so that we have representations for both prefix and tail)
        #
        self.logger.info('Getting possible splits')
        splits = set()

        for prefix in prefixes:
            rest = inputCompound[len(prefix):]

            # get all possible tails
            possible_tails = []  # (haus, 4)
            for fug in self.FUGENLAUTE:
                if rest.startswith(fug):
                    tail_offset = len(prefix) + len(fug)
                    possible_tails += [
                            (rest[len(fug):], tail_offset, fug),         #original case
                            (rest[len(fug):].title(), tail_offset, fug)  #title case
                    ]

            for (tail, tail_offset, fug) in possible_tails:
                self.logger.debug('Tail: %s, Fug: %s' % (tail, fug))
                if tail not in self.model.vocab:  # we haven't representation for this tail
                    self.logger.debug('Discarding split %s %s %s' % (inputCompound, prefix, tail))
                    continue
                splits.add((prefix, tail, tail_offset, fug))
                self.logger.debug('Considering split %s %s %s' % (inputCompound, prefix, tail))

        if len(splits) == 0:
            self.logger.error('Cannot decompound %s' % inputCompound)
            return []

        # 4. See if retrieved splits are good in terms of word embeddings
        #
        result = []
        self.logger.info('Applying direction vectors to possible splits')

        for prefix, tail, tail_offset, fug in splits:
            self.logger.debug('Applying %d directions vectors to split %s %s' %
                    (len(self.prototypes[prefix]), prefix, tail))

            for origin, evidence in self.prototypes[prefix]:
                self.logger.debug('Prefix %s by indexes %d and %d' % (prefix, origin[0], origin[1]))

                dirVectorCompoundRepresentation = self.model[self.model.index2word[origin[0]]]
                dirVectorTailRepresentation = self.model[self.model.index2word[origin[1]]]
                dirVectorDifference = dirVectorCompoundRepresentation - dirVectorTailRepresentation

                predictionRepresentation = self.model[tail] + dirVectorDifference

                self.logger.debug('Getting Annoy KNN')
                try:
                    neighbours = self.annoy_tree.get_nns_by_vector(list(predictionRepresentation),
                            self.globalNN)[:self.nAccuracy]
                    self.logger.debug(neighbours)
                except Exception as e:
                    print e
                    self.logger.error('Problem found when retrieving KNN for prediction representation')
                    self.logger.error(list(predictionRepresentation))
                    exit()

                # Find rank
                rank = -1
                for i, nei in enumerate(neighbours):
                    if nei == inputCompoundIndex:
                        rank = i
                if rank == -1:
                    self.logger.debug('%d not found in neighbours. NO RANK. WONT SPLIT' % inputCompoundIndex)
                    continue
                self.logger.debug('%d found in neighbours. Rank: %d' % (inputCompoundIndex, rank))

                # compare cosine against threshold
                similarity = cosine_similarity(predictionRepresentation, inputCompoundRep)[0][0]
                self.logger.debug('Computed cosine similarity: %f' % similarity)
                if similarity < self.similarityThreshold:
                    self.logger.debug('%d has too small cosine similarity, discarding' % inputCompoundIndex)
                    continue

                result.append((prefix, tail, tail_offset, origin[0],
                    origin[1], rank, similarity, fug))

        return result


    def get_decompound_lattice(self, inputCompound):
        # 1. Initialize
        #
        lattice = {}  # from: from -> (to, label, rank, cosine, fug)

        # 2. Make graph
        #
        def add_edges(from_, label):
            candidates = self.decompound(label)
            tails = set()

            #Default path:
            if from_ not in lattice:
                lattice[from_] = []

            lattice[from_] += [(from_, from_ + len(label), label, 0, 1.0, "")]

            #Canidate pathes:
            for index, candidate in enumerate(candidates):
                prefix, tail, tail_offset, origin0, origin1, rank, similarity, fug = candidate

                to = from_ + tail_offset
                lattice[from_] += [(from_, to, prefix, rank, similarity, fug)]

                tails.add((tail, tail_offset))
            for (tail, next_tail_offset) in tails:
                add_edges(from_ + next_tail_offset, tail)

        add_edges(0, inputCompound)

        for k in lattice.keys():
            lattice[k] = list(set(lattice[k]))

        return lattice

def print_path(viterbi_path):
    return " ".join(map(lambda p: "%d,%d,%s" % p, viterbi_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Decompound words.')

    parser.add_argument('model_folder')

    parser.add_argument('--mode', choices=['1-best', 'lattices', 'dict_w2v', 'w2v_dict'], default='1-best')
    parser.add_argument('--globalNN', default=500)
    parser.add_argument('--nAccuracy', default=250)
    parser.add_argument('--similarityThreshold', default=0.0)
    parser.add_argument('--prototypeFile', default="prototypes.p")

    args = parser.parse_args()

    print >> sys.stderr, "Loading model..."
    modelSetup = yaml.load(open(args.model_folder + "/model.yaml", 'r'))

    base_decompounder = BaseDecompounder(args.model_folder, modelSetup,
            nAccuracy=args.nAccuracy, globalNN=args.globalNN,
            similarityThreshold=args.similarityThreshold,
            prototype_file=args.prototypeFile)

    if args.mode == "lattices":
        for line in sys.stdin:
            print(
                base_decompounder.get_decompound_lattice(
                    line.decode('utf8').rstrip('\n').title(),
                )
            )
    elif args.mode == "w2v_dict":
        for word in base_decompounder.model.vocab.keys():
            print word.encode('utf-8')
    elif args.mode in ["1-best", "dict_w2v"]:
        vit = ViterbiDecompounder()
        vit.load_weights(modelSetup["WEIGHTS"])

        words = []
        if args.mode == "1-best":
            words = map(lambda line: line.decode('utf8').strip(),
                    sys.stdin)
        else:
            words = base_decompounder.model.vocab.keys()

        print >>sys.stderr, "# words: %d" % len(words)

        def process_word(word):
            lattice = Lattice(base_decompounder.get_decompound_lattice(word))
            viterbi_path = vit.viterbi_decode(Compound(word, None, lattice))
            return [word.encode('utf-8'), print_path(viterbi_path).encode('utf-8')]

        pool = multiprocessing.Pool(4)
        for pword in pool.map(process_word, words):
            print " ".join(pword)

