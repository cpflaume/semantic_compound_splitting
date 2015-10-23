#!/bin/bash
#PARAMS: PROTOTYPE_FILE OUT_FOLDER IN_FOLDER


export COMPOUND_MODEL="/home/jdaiber/compounds/de.model"
export PYTHON="/home/jdaiber/compounds/semantic_compound_splitting/venv/bin/python"
export CODE_DIR="/home/jdaiber/compounds_dict/semantic_compound_splitting"

if [ ! -d "$IN_FOLDER" ]; then
  mkdir $IN_FOLDER
  $PYTHON decompound.py /tmp/compound_model/ --mode w2v_dict > de.words
  split -l 32000 de.words $IN_FOLDER/words
fi

export IN_FOLDER=$IN_FOLDER
export OUT_FOLDER=$OUT_FOLDER
export PROTOTYPE_FILE=$PROTOTYPE_FILE

mkdir $OUT_FOLDER || echo "Folder exists"
all_jobs=""

for p in $(ls $IN_FOLDER/* | grep -v dict); do
    if [[ -s "$OUT_FOLDER/${p/$IN_FOLDER\//}.dict" ]]; then
        :
    else
        all_jobs="$all_jobs $p"
    fi
done

for p in $all_jobs; do
    qsub -v PROTOTYPE_FILE,OUT_FOLDER,CODE_DIR,PYTHON,COMPOUND_MODEL,IN_FOLDER,PART=${p/$IN_FOLDER\//} jobs/generate_dict.sh
done

#cat $OUT_FOLDER/*.dict | awk 'NF > 2' | sort > models/de.dict
