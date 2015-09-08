# shell for the job:
#PBS -S /bin/bash
#PBS -lnodes=1:mem64gb:ppn=15
#PBS -lwalltime=12:00:00

CODE_DIR=~/compounds/semantic_compound_splitting

cd $CODE_DIR

# venv
module load python 2>err_py
source venv/bin/activate 2>err_venv

# call the programs
python train_word2vec.py ~/compounds/plain_text/*.txt ~/compounds/out

