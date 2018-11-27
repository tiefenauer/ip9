#!/usr/bin/env bash

gpu=$1
if [[ ${gpu} = '' ]]; then
    echo "Enter GPU # to use for inference"
    read gpu
fi
echo "using GPU #${gpu} for all inferences!"

me=`basename "$0"`
target_dir="/media/D1/daniel.tiefenauer/performance_ls_en"

echo ${me} > ${target_dir}/${me}.log

cd ./src/
python3 evaluate_pipeline_en.py \
    --corpus /media/D1/daniel.tiefenauer/corpora/librispeech \
    --language en \
    --keras_path /media/D1/daniel.tiefenauer/_runs/lc_cv_en_ds_dropouts/1000_min \
    --ds_path /media/D1/daniel.tiefenauer/asr/model/output_graph.pbmm \
    --ds_alpha_path /media/D1/daniel.tiefenauer/asr/alphabet.txt \
    --ds_trie_path /media/D1/daniel.tiefenauer/asr/model/trie \
    --lm_path /media/D1/daniel.tiefenauer/asr/lm.binary \
    --gpu ${gpu} \
    --target_dir ${target_dir} | tee ${target_dir}/${me}.log