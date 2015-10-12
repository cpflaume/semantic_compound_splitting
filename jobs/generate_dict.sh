# shell for the job:
#PBS -S /bin/bash
#PBS -lnodes=1:mem64gb:ppn=15
#PBS -lwalltime=12:00:00

cd $CODE_DIR

# venv
module load python 2>err_py

# call the programs
$PYTHON decompound.py $COMPOUND_MODEL < $PART > ${PART}.dict

