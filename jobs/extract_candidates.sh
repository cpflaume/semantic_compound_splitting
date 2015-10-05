# shell for the job:
#PBS -S /bin/bash
#PBS -lnodes=1:cores16:mem64gb
#PBS -lwalltime=48:00:00

cd $CODE_DIR

# venv
module load python  2>err_py

$PYTHON training/extract_candidates.py -w $OUT_DIR/w2v.bin -b $OUT_DIR/dawg -c $OUT_DIR/candidates -o $OUT_DIR/annoy_index -i $OUT_DIR/candidates_index.p -l $MIN_LENGTH -n 100 -f "$FUGENELEMENTE"

touch $OUT_DIR/step2.done
