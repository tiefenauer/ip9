#!/usr/bin/env bash

me=`basename "$0"`
target_dir="/media/D1/daniel.tiefenauer/_performance_$me"

echo ${me} > ${target_dir}/${me}.log

cd ./src/
python3 evaluate_pipeline.py \
    --corpus /media/D1/daniel.tiefenauer/corpora/readylingua \
    --language en \
    --keras_path /media/D1/daniel.tiefenauer/_runs/lc_cv_en_ds/1000_min \
    --ds_path /media/D1/daniel.tiefenauer/asr/model/output_graph.pbmm \
    --ds_alpha_path /media/D1/daniel.tiefenauer/asr/model/alphabet.txt \
    --/media/D1/daniel.tiefenauer/asr/model/trie \
    --lm_path /media/D1/daniel.tiefenauer/ds_en/lm.binary \
    --target_dir ${target_dir}| tee ${target_dir}/${me}.log