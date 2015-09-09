# shell for the job:
#PBS -S /bin/bash
#PBS -lnodes=1:mem64gb:ppn=15
#PBS -lwalltime=12:00:00

CODE_DIR=~/compounds/semantic_compound_splitting
CMPD_DIR=/home/jdaiber1/compounds/

cd $CODE_DIR

# venv
module load python 2>err_py

# call the programs
venv/bin/python train_word2vec.py "$CMPD_DIR/plain_text/*.txt" $CMPD_DIR/out/w2v.bin 2>&1 > $CMPD_DIR/out/word2vec.log

