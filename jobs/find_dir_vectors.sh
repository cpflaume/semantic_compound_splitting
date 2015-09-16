# shell for the job:
#PBS -S /bin/bash
# use one node with 16 cores:
#PBS -lnodes=1:cores16
# job requires at most 4 hours, 0 minutes
#     and 0 seconds wallclock time
#PBS -lwalltime=12:00:00

cd $CODE_DIR

# venv
module load python 2>> err_py

# call the programs
$PYTHON training/find_direction_vectors.py -w $OUT_DIR/w2v.bin -d 500 -t $OUT_DIR/annoy_index -c $OUT_DIR/$INPUT -o $OUT_DIR/prototypes/$OUTPUT -p 16 -s 500 -r 80 -e 6 > $OUT_DIR/prototypes/log/${OUTPUT}.log 2>&1


