#!/bin/bash

PRED=$1
GT=$2
OUTPUT=$3
CONFIG=$4

python versa/bin/scorer.py \
    --pred ${PRED} \
    --gt ${GT} \
     --io soundfile \
    --output_file ${OUTPUT} \
    --score_config ${CONFIG}
