#!/usr/bin/env bash

me=`basename "$0"`
target_dir="/media/D1/daniel.tiefenauer/performance_rl_de"

mkdir -p ${target_dir}
echo ${me} > ${target_dir}/${me}.log

cd ./src/
python3 evaluate_pipeline_de.py \
    --corpus rl \
    --language de \
    --keras_path /home/daniel/Documents/_runs/lc_rl_de_wiki_synth/1000_min \
    --target_dir /media/daniel/IP9/demos/performance_rl_de | tee ${target_dir}/${me}.log