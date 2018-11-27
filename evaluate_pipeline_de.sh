#!/usr/bin/env bash

gpu=$1
if [[ ${gpu} = '' ]]; then
    echo "Enter GPU # to use for inference"
    read gpu
fi
echo "using GPU #${gpu} for all inferences!"

me=`basename "$0"`
target_dir="/media/D1/daniel.tiefenauer/performance_rl_de"

mkdir -p ${target_dir}
echo ${me} > ${target_dir}/${me}.log

cd ./src/
python3 evaluate_pipeline_de.py \
    --keras_path /media/D1/daniel.tiefenauer/_runs/lc_rl_de_wiki_synth/1000_min \
    --gpu ${gpu} \
    --target_dir /media/D1/daniel.tiefenauer/performance_rl_de | tee ${target_dir}/${me}.log