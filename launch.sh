#!/bin/bash

PRED_WAVSCP=$1
GT_WAVSCP=$2
SCORE_DIR=$3
SPLIT_SIZE=$4

GPU_PART=general
CPU_PART=general


mkdir -p ${SCORE_DIR}
mkdir -p ${SCORE_DIR}/pred
mkdir -p ${SCORE_DIR}/gt
mkdir -p ${SCORE_DIR}/result

total_lines=$(wc -l < "${PRED_WAVSCP}")
source_wavscp=$(basename ${PRED_WAVSCP})
lines_per_piece=$(( (total_lines + SPLIT_SIZE) / SPLIT_SIZE ))
split -l ${lines_per_piece} -d -a 3 ${PRED_WAVSCP} ${SCORE_DIR}/pred/${source_wavscp}_
pred_list=(${SCORE_DIR}/pred/${source_wavscp}_*)

if [ "${GT_WAVSCP}" = None ]; then
    echo "No ground truth audio received, skipping ground truth info"
else
    target_wavscp=$(basename ${GT_WAVSCP})
    split -l ${lines_per_piece} -d -a 3 ${GT_WAVSCP} ${SCORE_DIR}/gt/${target_wavscp}_
    gt_list=(${SCORE_DIR}/gt/${target_wavscp}_*)

    if [ ${#pred_list[@]} -ne ${#gt_list[@]} ]; then
        echo "The number of split ground truth and predictions does not match."
        exit 1
    fi
fi



for ((i=0; i<${#pred_list[@]}; i++))  ; do

    sub_pred_wavscp=${pred_list[${i}]}

    if [ "${GT_WAVSCP}" = None ]; then
        sub_gt_wavscp=None
    else
        sub_gt_wavscp=${gt_list[${i}]}
    fi
    
    echo "${sub_pred_wavscp} ${sub_gt_wavscp}"

    sbatch \
        -p ${GPU_PART} \
        --time 2-0:00:00 \
        --cpus-per-task 8 \
        --mem-per-cpu 2000M \
        --gres=gpu:A6000:1 \
        ./egs/run_gpu.sh \
            ${sub_pred_wavscp} \
            ${sub_gt_wavscp} \
            ${SCORE_DIR}/result/$(basename ${sub_pred_wavscp}).result.gpu.txt \
            egs/speech_gpu.yaml

    sbatch \
        -p ${CPU_PART} \
        --time 2-0:00:00 \
        --cpus-per-task 8 \
        --mem-per-cpu 2000M \
        ./egs/run_cpu.sh \
            ${sub_pred_wavscp} \
            ${sub_gt_wavscp} \
            ${SCORE_DIR}/result/$(basename ${sub_pred_wavscp}).result.cpu.txt \
            egs/speech_cpu.yaml

done


