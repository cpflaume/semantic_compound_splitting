import gensim
import sys
import glob

import codecs
from nltk.tokenize import RegexpTokenizer
import glob
import sys


class CorpusReader():
    """
    Reads corpus from gzip file.
    """

    def __init__(self, files):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self.tokenizer = RegexpTokenizer(r'\w+')

    def __iter__(self):
        """
        Generator that returns a list of tokens for each sentence.
        :return: list of tokens
        """
        for f in self.files:
            print("Processing ", f)
            for line in open(f, "r"):
		try:
	                yield self.tokenizer.tokenize(line)
		except:
			pass

print("Starting W2V training...")

files = glob.glob(sys.argv[1])
outfile_name = sys.argv[2]

dataset = CorpusReader(files)
model = gensim.models.Word2Vec(dataset, size=500, window=5, min_count=3, negative=5, workers=15)

model.save(outfile_name)

