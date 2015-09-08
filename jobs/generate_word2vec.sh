# shell for the job:
#PBS -S /bin/bash
#PBS -lnodes=1:cores16:mem64gb
#PBS -lwalltime=00:40:00

CODE_DIR=~/compounds/semantic_compound_splitting

cd $CODE_DIR

# venv
module load python 2>err_py
source venv/bin/activate 2>err_venv

# call the programs
python train_word2vec.py ~/compounds/plain_text ~/compounds/out


