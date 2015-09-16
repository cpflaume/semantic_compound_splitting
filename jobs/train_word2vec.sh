# shell for the job:
#PBS -S /bin/bash
#PBS -lnodes=1:mem64gb:ppn=15
#PBS -lwalltime=12:00:00

cd $CODE_DIR

# venv
module load python 2>err_py

# call the programs
$PYTHON training/train_word2vec.py "$INPUT_TEXTS" $OUT_DIR/w2v.bin 2>&1 > $OUT_DIR/word2vec.log

touch $OUT_DIR/step1.done
