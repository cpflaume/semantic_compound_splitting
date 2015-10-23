#PBS -S /bin/bash
#PBS -lnodes=1:mem64gb:ppn=15
#PBS -lwalltime=00:60:00

module load python 2>err_py

echo $PYTHON
echo $CODE_DIR
cd $CODE_DIR

echo "In folder: $IN_FOLDER"
echo "Out folder: $OUT_FOLDER"

# call the programs
$PYTHON decompound.py $COMPOUND_MODEL --prototypeFile $PROTOTYPE_FILE < $IN_FOLDER/$PART > $OUT_FOLDER/${PART}.dict

