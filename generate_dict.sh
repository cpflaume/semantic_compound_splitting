#!/bin/bash

export COMPOUND_MODEL="/home/jdaiber/compounds/de.model"
export PYTHON="/home/jdaiber/compounds/semantic_compound_splitting/venv/bin/python"
export CODE_DIR="/home/jdaiber/compounds_dict/semantic_compound_splitting"

#$PYTHON decompound.py /tmp/compound_model/ --mode w2v_dict > de.words

#mkdir de_words
#split -l 32000 de.words de_words/words

all_jobs=""

for p in $(ls de_words/* | grep -v dict); do
    if [[ -s "${p}.dict" ]]; then
        :
    else
        all_jobs="$all_jobs $p"
    fi
done

for p in $all_jobs; do
    qsub -v CODE_DIR,PYTHON,COMPOUND_MODEL,PART=$p jobs/generate_dict.sh
done

#cat de_words/*.dict | awk 'NF > 2' | sort > models/de.dict
