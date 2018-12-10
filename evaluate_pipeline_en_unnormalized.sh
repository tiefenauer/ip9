#!/usr/bin/env bash

gpu=$1
if [[ ${gpu} = '' ]]; then
    echo "Enter GPU # to use for inference"
    read gpu
fi
echo "using GPU #${gpu} for all inferences!"

echo "Force re-alignment? (Y/n)"
read force_realignment
for v in "N" "n"
do
    if [[ "$force_realignment" = ${v} ]]; then
        force_realignment=false
    fi
done

if [[ "$force_realignment" = false ]]; then
    force_realignment=""
else
    echo "Forcing re-alignment"
    force_realignment="--force_realignment"
fi


me=`basename "$0"`
target_dir="/media/D1/daniel.tiefenauer/performance_ls_en_unnormalized"

echo ${me} > ${target_dir}/${me}.log

cd ./src/
python3 evaluate_pipeline_en.py \
    --source_dir /media/D1/daniel.tiefenauer/corpora/librispeech-test-unnormalized \
    --language en \
    --keras_path /media/D1/daniel.tiefenauer/_runs/lc_cv_en_ds_dropouts/1000_min \
    --ds_path /media/D1/daniel.tiefenauer/asr/output_graph.pbmm \
    --ds_alpha_path /media/D1/daniel.tiefenauer/asr/alphabet.txt \
    --ds_trie_path /media/D1/daniel.tiefenauer/asr/trie \
    --lm_path /media/D1/daniel.tiefenauer/asr/lm.binary \
    ${force_realignment} \
    --align_endings \
    --gpu ${gpu} \
    --target_dir ${target_dir} | tee ${target_dir}/${me}.log