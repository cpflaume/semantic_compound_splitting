import cPickle as pickle
import sys
import numpy as np

# argument1=candidate_index_file, argument2=number_of_splits,
# argument3 = minimum evidence length

candidates = pickle.load(open(sys.argv[1], "rb"))

minimum_length = int(sys.argv[3])

length_tuples = sorted([(c, min(500,len(candidates[c]))) for c in candidates if len(candidates[c]) >= minimum_length], key=lambda t: t[1], reverse=True)

print "Number of valid rules: ", len(length_tuples)

num_splits = int(sys.argv[2])

splits = [{} for i in range(num_splits)]
lengths = [0 for i in range(num_splits)]

# Simple heuristic

for prefix, length in length_tuples:
    target = np.argmin(lengths)
    splits[target][prefix] = candidates[prefix]
    lengths[target] += length**2

print "Length distribution:", lengths

for i, split in enumerate(splits):
    pickle.dump(split, open(sys.argv[1]+"."+str(i+1), "wb"))
