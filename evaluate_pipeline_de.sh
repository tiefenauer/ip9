#!/usr/bin/env bash

gpu=$1
if [[ ${gpu} = '' ]]; then
    echo "Enter GPU # to use for inference"
    read gpu
fi
echo "using GPU #${gpu} for all inferences!"

echo "Force re-alignment? (Y/n)"
read force_realignment
for v in "N" "n" do
    if [[ "$force_realignment" = v ]]; then
        force_realignment=false
    fi
do
done

if [[ "$force_realignment" = false ]]; then
    force_realignment=""
else
    force_realignment="--force_realignment"
fi


me=`basename "$0"`
target_dir="/media/D1/daniel.tiefenauer/performance_rl_de"

mkdir -p ${target_dir}
echo ${me} > ${target_dir}/${me}.log

cd ./src/
python3 evaluate_pipeline_de.py \
    --corpus /media/D1/daniel.tiefenauer/corpora/readylingua \
    --keras_path /media/D1/daniel.tiefenauer/_runs/lc_rl_de_wiki_synth/1000_min \
    ${force_realignment}
    --gpu ${gpu} \
    --target_dir ${target_dir} | tee ${target_dir}/${me}.log