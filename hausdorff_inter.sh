#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ] ; then
    >&2 echo 'No arguments supplied'
    echo "USAGE:"
    echo ">> $0 USER0 USER1 PATIENT OUTPUT_DIR mesh0 mesh1"
    echo ""
    echo "Output saved at: OUTPUT_DIR/PATIENT_Hausdorff.vtk"
    echo "                 OUTPUT_DIR/PATIENT_Distance.dat"
    exit 1
fi

SCRIPT_DIR=$( cd -- $( dirname -- ${BASH_SOURCE[0]}  ) &> /dev/null && pwd )


DIR0="$AFIB_REPROD/$1/03_completed/$3"
DIR1="$AFIB_REPROD/$2/03_completed/$3"
DIRo=$4

python $SCRIPT_DIR/calc_hausdorff.py $DIR0 $DIR1 $DIRo $5 $6 "$3_Hausdorff"
