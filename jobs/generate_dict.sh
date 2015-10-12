#PBS -S /bin/bash
#PBS -lnodes=1:mem64gb:ppn=15
#PBS -lwalltime=00:20:00

module load python 2>err_py

echo $PYTHON
echo $CODE_DIR
cd $CODE_DIR

echo "In folder: $PWD"

# call the programs
$PYTHON decompound.py $COMPOUND_MODEL < $PART > ${PART}.dict

