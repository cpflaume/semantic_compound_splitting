#!/bin/bash

export COMPOUND_MODEL="/home/jdaiber/compounds/de.model"
export PYTHON="/home/jdaiber/compounds/semantic_compound_splitting/venv/bin/python"
export CODE_DIR="/home/jdaiber/compounds_dict/semantic_compound_splitting"

#python decompound.py /tmp/compound_model/ --mode w2v_dict > de.words
#mkdir de_words
#split -l 32000 de.words de_words/words

for p in de_words/*; do
    qsub -v CODE_DIR,PYTHON,COMPOUND_MODEL,PART=$p jobs/generate_dict.sh
done

#cat de_words/*.dict > de.dict
