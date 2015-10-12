#!/bin/bash

COMPOUND_MODEL=""
PYTHON="python"
CODE_DIR=""

#python decompound.py /tmp/compound_model/ --mode w2v_dict > de.words
#mkdir de_words
#split -l 32000 de.words de_words/words

for $p in de_words/*; do
    qsub -v CODE_DIR="..." -v PYTHON="..." COMPOUND_MODEL="$COMPOUND_MODEL" PART="$p"
done

#cat de_words/*.dict > de.dict
