# shell for the job:
#PBS -S /bin/bash
#PBS -lnodes=1:mem64gb:ppn=15
#PBS -lwalltime=12:00:00

CODE_DIR=~/compounds/semantic_compound_splitting

cd $CODE_DIR

# venv
module load python 2>err_py

# call the programs
venv/bin/python train_word2vec.py ~/compounds/plain_text/*.txt ~/compounds/out/w2v.bin 2>&1 > ~/compounds/out/word2vec.log

