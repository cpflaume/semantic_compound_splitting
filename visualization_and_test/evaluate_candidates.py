__author__ = 'rwechsler'
import pickle as pickle
import itertools
import random
from annoy import AnnoyIndex
import multiprocessing as mp
import sys
import argparse
import time
import datetime
import numpy as np
import gensim



def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def load_candidate_dump(file_name):
    return pickle.load(open(file_name, "rb"))

def load_annoy_tree(model_file_name, vector_dims):
    tree = AnnoyIndex(vector_dims)
    tree.load(model_file_name)
    return tree

def load_word2vecmodel(file_name):
    return gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)

def annoy_knn(annoy_tree, vector, true_index, k=100):
    neighbours = annoy_tree.get_nns_by_vector(list(vector), k)
    if true_index in neighbours:
        return True
    else:
        return False

def test_pair(pair1, pair2, word2vec_model, k=100, show=30):
    """
    Only used in interactive mode so far.
    :param pair1:
    :param pair2:
    :param word2vec_model:
    :param k:
    :param show:
    :return:
    """
    prefix = pair1[0]
    fl1 = pair1[1]
    tail1 = pair1[2]
    prefix2 = pair2[0]
    fl2 = pair2[1]
    tail2 = pair2[2]
    assert prefix == prefix2

    diff = word2vec_model[prefix + fl2 + tail2.lower()] - word2vec_model[tail2]
    predicted = word2vec_model[tail1] + diff

    true_word = prefix + fl1 + tail1.lower()

    neighbours = word2vec_model.most_similar([predicted], topn=k)

    print(neighbours[:show])
    neighbours, _ = list(zip(*neighbours))
    print("Found: ", true_word in neighbours)


def candidate_generator(candidates, rank_threshold):
    for prefix in candidates:
        yield (prefix, candidates[prefix], rank_threshold)

def mp_wrapper_evaluate_set(argument):
    return evaluate_set(*argument)

if __name__ == "__main__":


    #### Default Parameters-------------------------------------------####
    rank_threshold = 100
    sample_set_size = 500
    n_processes = 2
    ####End-Parametes-------------------------------------------------####


    parser = argparse.ArgumentParser(description='Evaluate candidates')
    parser.add_argument('-w', action='store', dest="word2vec_file", required=True)
    parser.add_argument('-d', action="store", dest="vector_dims", type=int, required=True)
    parser.add_argument('-t', action="store", dest="annoy_tree_file", required=True)
    parser.add_argument('-c', action="store", dest="candidates_index_file", required=True)
    parser.add_argument('-o', action="store", dest="result_output_file", required=True)
    parser.add_argument('-p', action="store", dest="n_processes", type=int, default=n_processes)
    parser.add_argument('-s', action="store", dest="sample_set_size", type=int, default=sample_set_size)
    parser.add_argument('-r', action="store", dest="rank_threshold", type=int, default=rank_threshold)

    arguments = parser.parse_args(sys.argv[1:])


    print(timestamp(), "loading candidates")
    candidates = load_candidate_dump(arguments.candidates_index_file)

    print(timestamp(), "loading word2vec model")
    word2vec_model = load_word2vecmodel(arguments.word2vec_file)

    print(timestamp(), "preprocess candidates")
    # only store vectors that we need. And sample already.
    word2vec_vectors = dict()
    for cand in candidates:
        if len(candidates[cand]) > arguments.sample_set_size:
            candidates[cand] = set(random.sample(candidates[cand], arguments.sample_set_size))
        for (i,j) in candidates[cand]:
            word2vec_vectors[i] = np.array(word2vec_model.syn0[i])
            word2vec_vectors[j] = np.array(word2vec_model.syn0[j])

    del word2vec_model

    print(timestamp(), "number of vectors: ", len(word2vec_vectors))

    print(timestamp(), "load annoy tree")
    # global annoy_tree
    annoy_tree = load_annoy_tree(arguments.annoy_tree_file, arguments.vector_dims)

    def evaluate_set(prefix, tails, rank_threshold=100):
        global annoy_tree
        global word2vec_vectors

        counts = dict()
        counts[True] = 0
        counts[False] = 0

        #print mp.current_process().name, id(annoy_tree), id(word2vec_vectors), prefix.encode('utf-8')
        #sys.stdout.flush()

        for (comp1, tail1), (comp2, tail2) in itertools.combinations(tails, 2):

            diff = word2vec_vectors[comp2]- word2vec_vectors[tail2]
            predicted = word2vec_vectors[tail1] + diff

            result = annoy_knn(annoy_tree, predicted, comp1, rank_threshold)

            counts[result] += 1

        return (prefix, float(counts[True]) / (counts[True] + counts[False])) if counts[True] + counts[False] > 0 else (prefix, 0.0)

    print(timestamp(), "evaluating candidates")
    pool = mp.Pool(processes=arguments.n_processes)
    params = candidate_generator(candidates, arguments.rank_threshold)
    results = pool.map(mp_wrapper_evaluate_set, params)

    print(timestamp(), "pickling")
    pickle.dump(results, open(arguments.result_output_file, "wb"))

    print(timestamp(), "done")
