# shell for the job:
#PBS -S /bin/bash
#PBS -lnodes=1:cores16:mem64gb
#PBS -lwalltime=48:00:00

# cd to the directory where the program is to be called:

# venv
module load python  2>err_py
source venv/bin/activate 2>err_venv

python 
