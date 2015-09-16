#!/usr/bin/env python3
import argparse
import fileinput
from compound import Compound
from lattice import Lattice

# import cPickle as pickle
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

from viterbi_decompounder import ViterbiDecompounder


def decompound(inputCompound, nAccuracy, similarityThreshold, offset=0):
    global annoy_tree
    global prototypes
    global model
    global globalNN

    # 1. See if we can deal with compound
    #
    if len(inputCompound) == 0:
        return []

    logger.debug('Looking up word %s in Word2Vec model' % inputCompound)
    if inputCompound not in model.vocab:  # We haven't vector representation for compound
        logger.debug('ERROR COULDNT FIND KEY %s IN WORD2VEC MODEL' % inputCompound)
        return []
    logger.debug('Found key in index dict for word %s' % inputCompound)
    inputCompoundRep = model[inputCompound]
    inputCompoundIndex = model.vocab[inputCompound].index

    # 2. See if we have prefixes of compound in vocabulary
    #
    logger.info('Getting all matching prefixes')
    prefixes = set()
    for prefix in prototypes.keys():
        if len(inputCompound) > len(prefix) and inputCompound.startswith(prefix):
            prefixes.add(prefix)
    logger.debug('Possible prefixes: %r' % prefixes)
    if len(prefixes) == 0:  # cannot split
        return []

    # 3. Get all possible splits (so that we have representations for both prefix and tail)
    #
    logger.info('Getting possible splits')
    splits = set()

    FUGENLAUTE = ['', 'e', 'es']  # Needs to include '' !!!
    for prefix in prefixes:
        rest = inputCompound[len(prefix):]

        # get all possible tails
        possible_tails = []  # (haus, 4)
        for fug in FUGENLAUTE:
            if rest.startswith(fug):
                tail_offset = len(prefix) + len(fug)
                possible_tails += [(rest[len(fug):], tail_offset), (rest[len(fug):].title(), tail_offset)]

        for (tail, tail_offset) in possible_tails:
            logger.debug('Tail: %s' % tail)
            if tail not in model.vocab:  # we haven't representation for this tail
                logger.debug('Discarding split %s %s %s' % (inputCompound, prefix, tail))
                continue
            splits.add((prefix, tail, tail_offset))
            logger.debug('Considering split %s %s %s' % (inputCompound, prefix, tail))

    if len(splits) == 0:
        logger.error('Cannot decompound %s' % inputCompound)
        return []

    # 4. See if retrieved splits are good in terms of word embeddings
    #
    result = []
    logger.info('Applying direction vectors to possible splits')

    for prefix, tail, tail_offset in splits:
        logger.debug('Applying %d directions vectors to split %s %s' % (len(prototypes[prefix]), prefix, tail))

        for origin, evidence in prototypes[prefix]:
            logger.debug('Prefix %s by indexes %d and %d' % (prefix, origin[0], origin[1]))

            dirVectorCompoundRepresentation = model[model.index2word[origin[0]]]
            dirVectorTailRepresentation = model[model.index2word[origin[1]]]
            dirVectorDifference = dirVectorCompoundRepresentation - dirVectorTailRepresentation

            predictionRepresentation = model[tail] + dirVectorDifference

            logger.debug('Getting Annoy KNN')
            try:
                neighbours = annoy_tree.get_nns_by_vector(list(predictionRepresentation), globalNN)[:nAccuracy]
                logger.debug(neighbours)
            except:
                logger.error('Problem found when retrieving KNN for prediction representation')
                logger.error(list(predictionRepresentation))
                exit()

            # Find rank
            rank = -1
            for i, nei in enumerate(neighbours):
                if nei == inputCompoundIndex:
                    rank = i
            if rank == -1:
                logger.debug('%d not found in neighbours. NO RANK. WONT SPLIT' % inputCompoundIndex)
                continue
            logger.debug('%d found in neighbours. Rank: %d' % (inputCompoundIndex, rank))

            # compare cosine against threshold
            similarity = cosine_similarity(predictionRepresentation, inputCompoundRep)[0][0]
            logger.debug('Computed cosine similarity: %f' % similarity)
            if similarity < similarityThreshold:
                logger.debug('%d has too small cosine similarity, discarding' % inputCompoundIndex)
                continue

            result.append((prefix, tail, tail_offset, origin[0], origin[1], rank, similarity))

    return result


vertices_count = 0
distances = []


def get_decompound_lattice(inputCompound, nAccuracy, similarityThreshold):
    # 1. Initialize
    #
    lattice = {}  # from: from -> (to, label, rank, cosine)

    # 2. Make graph
    #
    def add_edges(from_, label):
        candidates = decompound(label, nAccuracy, similarityThreshold)
        tails = set()

        lattice[from_] = [(from_, from_ + len(label), label, 0, 1.0)]
        for index, candidate in enumerate(candidates):
            prefix, tail, tail_offset, origin0, origin1, rank, similarity = candidate

            to = from_ + tail_offset
            lattice[from_] += [(from_, to, prefix, rank, similarity)]

            tails.add((tail, tail_offset))

        for (tail, next_tail_offset) in tails:
            add_edges(from_ + next_tail_offset, tail)

    add_edges(0, inputCompound)

    return lattice


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Decompound words.')

    parser.add_argument('model_folder')

    parser.add_argument('mode', choices=['1-best', 'lattices'], default='1-best')
    parser.add_argument('--globalNN', default=500)
    parser.add_argument('--nAccuracy', default=250)
    parser.add_argument('--similarityThreshold', default=0.0)
    parser.add_argument('--weightsFile', default='weights')

    args = parser.parse_args()

    # Basic Logging:
    logger = logging.getLogger('')
    hdlr = logging.FileHandler('decompound_annoy.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.CRITICAL)

    print >> sys.stderr, "Loading prototypes..."
    prototypes = pickle.load(open(args.model_folder + '/prototypes.p', 'rb'))

    print >> sys.stderr, "Loading KNN search..."
    annoy_tree = AnnoyIndex(500)
    annoy_tree.load(args.model_folder + '/tree.ann')

    print >> sys.stderr, "Loading gensim model..."
    model = gensim.models.Word2Vec.load_word2vec_format(args.model_folder + '/w2v.bin', binary=True)

    print >> sys.stderr, "Done."

    if args.mode == "lattices":
        for line in sys.stdin:
            print(get_decompound_lattice(line.decode('utf8').rstrip('\n').title(), args.nAccuracy, args.similarityThreshold))
    elif args.mode == "1-best":
        vit = ViterbiDecompounder()
        vit.load_weights(args.weightsFile)

        for line in fileinput.input():
            c = line.decode('utf8').strip()
            lattice = Lattice(get_decompound_lattice(c, args.nAccuracy, args.similarityThreshold))

            viterbi_path = vit.viterbi_decode(Compound(c, None, lattice))
            print " ".join(map(lambda p: "%d,%d" % p, viterbi_path))


