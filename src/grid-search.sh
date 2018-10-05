#!/usr/bin/env bash
# set -xe
usage="$(basename "$0") [-h|--help] [-r|--run_id <string>] [-d|--destination <path>] [-t|--train_files <path>] [-v|--valid_files <path>] [-g|--gpu] [-b|--batch_size <int>] [-e|--epochs <int>]
where:
    -h|--help                    show this help text
    -r|--run_id <string>         run-id to use (used to resume training)    
    -d|--destination <path>      destination directory to store results
    -l|--lm                      path to n-gram KenLM model (if possible binary)
    -a|--lm_vocab                path to file containing the vocabulary of the LM specified by -lm. The file must contain the words used for training delimited by space (no newlines)
    -t|--train_files <path>      one or more comma-separated paths to CSV files containing the corpus files to use for training
    -v|--valid_files <path>      one or more comma-separated paths to CSV files containing the corpus files to use for validation
    -g|--gpu <int>               GPU to use
    -b|--batch_size <int>        batch size
    -e|--epochs <int>            number of epochs to train

Create data to plot a learning curve by running a simplified version of the DeepSpeech-BRNN. The purpose of this script is simply to call ./run-train.sh with varying parameters with a gred-search along the following dimensions:

- time dimension: use increasing amounts of training data (1 to 1000 minutes)
- decoder dimension: use different decoding methods  (Beam Search, Best-Path and Old)
- LM dimension: train with or without a Language model (LM)

For each element in the cartesian product of these dimensions a training run is started. A unique run-id is assigned to each training run from which the value of each dimension can be derived.
"

# Defaults
gs_run_id="grid_search_$(uuidgen)"
lm=''
lm_vocab=''
train_files='/media/D1/readylingua-en/readylingua-en-train.csv'
valid_files='/media/D1/readylingua-en/readylingua-en-dev.csv'
target_dir='/home/daniel_tiefenauer/learning_curve_0'
gpu=''
batch_size='16'
epochs='20'

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
    echo ${usage}
    shift
    exit
    ;;
    -r|--run_id)
    gs_run_id="$2"
    shift
    shift
    ;;
    -d|--destination)
    target_dir="$2"
    shift
    shift
    ;;
    -l|--lm)
    lm="$2"
    shift
    shift
    ;;
    -a|--lm_vocab)
    lm_vocab="$2"
    shift
    shift
    ;;
    -t|--train_files)
    train_files="$2"
    shift
    shift
    ;;
    -v|--valid_files)
    valid_files="$2"
    shift
    shift
    ;;
    -g|--gpu)
    gpu="$2"
    shift
    shift
    ;;
    -b|--batch_size)
    batch_size="$2"
    shift
    shift
    ;;
    -e|--epochs)
    epochs="$2"
    shift
    shift
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

gs_result_dir=${target_dir%/}/${gs_run_id}
mkdir -p ${gs_result_dir}

echo "
-----------------------------------------------------
 starting learning curve with the following parameters
-----------------------------------------------------
gs_run_id     = ${gs_run_id}
gs_result_dir = ${gs_result_dir}
target_dir    = ${target_dir}
lm            = ${lm}
lm_vocab      = ${lm_vocab}
train_files   = ${train_files}
valid_files   = ${valid_files}
gpu           = ${gpu}
batch_size    = ${batch_size}
epochs        = ${epochs}
-----------------------------------------------------
" | tee ${gs_result_dir%/}/${gs_run_id}.log

# time dimension
for minutes in 1 10 100 1000
do
    # LM dimension
    for use_lm in true false
    do
        # decoder dimension
        for decoder in 'beamsearch' 'bestpath' 'old'
        do
            if [[${use_lm} == true]]; then
                lm_str="withLM"
            else
                lm_str="noLM"
                lm=''
                lm_vocab=''
            fi
            run_id="${minutes}_min_${lm_str}_${decoder}"
            target_subdir=${target_dir%/}/${run_id}

            mkdir -p ${target_subdir}

            echo "
            #################################################################################################
             Training on $minutes minutes, use_lm=$use_lm, decoding=$decoder
             run id: $run_id
             target subdirectory: $target_subdir
             LM: $lm
             LM vocab: $lm_vocab
            #################################################################################################
            "

            python3 run-train.py \
                    --run_id ${run_id} \
                    --target_dir ${target_dir} \
                    --minutes ${minutes} \
                    --decoder ${decoder} \
                    --lm ${lm} \
                    --lm_vocab ${lm_vocab} \
                    --gpu ${gpu} \
                    --batch_size ${batch_size} \
                    --epochs ${epochs} \
                    --train_files ${train_files} \
                    --valid_files ${valid_files} \
                    2>&1 | tee ${target_dir}/${run_id}.log # write to stdout and log file

            echo "
            #################################################################################################
             Finished $run_id
            #################################################################################################
            "
        done

    done

done
