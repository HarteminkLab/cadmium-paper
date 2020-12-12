#!/bin/bash

BED_FILE=${1}
OUTPUT_DIR=${2}
DANPOS_PATH=${3}

WORKING_DIR=$(pwd)

cd $OUTPUT_DIR/danpos/

echo "BED: $BED_FILE"
echo "OUTPUT: $OUTPUT_DIR"
echo "DANPOS: $DANPOS_PATH"

python $DANPOS_PATH dpos --paired 1 $BED_FILE
