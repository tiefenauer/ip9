#!/usr/bin/env bash
# set -xe
usage="$(basename "$0") [-h|--help] [-r|--run_id <string>] [-d|--destination <path>] [-t|--train_files <path>] [-v|--valid_files <path>] [-g|--gpu] [-b|--batch_size <int>] [-e|--epochs <int>] [-x|--decoder <string>] [-l|--lm <path>] [-a|--lm_vocab <path>]
where:
    -h|--help                                show this help text
    -r|--run_id <string>                     run-id to use (used to resume training)
    -d|--destination <path>                  destination directory to store results
    -l|--lm                                  path to n-gram KenLM model (if possible binary)
    -a|--lm_vocab                            path to file containing the vocabulary of the LM specified by -lm. The file must contain the words used for training delimited by space (no newlines)
    -x|--decoder <'beamsearch'|'bestpath'>   decoder to use (default: beamsearch)
    -t|--train_files <path>                  one or more comma-separated paths to CSV files containing the corpus files to use for training
    -v|--valid_files <path>                  one or more comma-separated paths to CSV files containing the corpus files to use for validation
    -g|--gpu <int>                           GPU to use (default: 2)
    -b|--batch_size <int>                    batch size
    -e|--epochs <int>                        number of epochs to train

Create data to plot a learning curve by running a simplified version of the DeepSpeech-BRNN. This script will call run-train.py with increasing amounts of training data (1 to 1000 minutes).
For each amount of training data a separate training run is started. A unique run-id is assigned to each training run from which the value of each dimension can be derived.
"

# Defaults
lc_run_id="learning_run_$(uuidgen)"
lm=''
lm_vocab=''
decoder='beamsearch'
train_files='/media/D1/readylingua-en/readylingua-en-train.csv'
valid_files='/media/D1/readylingua-en/readylingua-en-dev.csv'
target_dir='/home/daniel_tiefenauer/learning_curve_0'
gpu='2'
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
    lc_run_id="$2"
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
    -x|--decoder)
    decoder="$2"
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

lc_result_dir=${target_dir%/}/${lc_run_id}
mkdir -p ${lc_result_dir}

echo "
-----------------------------------------------------
 starting learning curve with the following parameters
-----------------------------------------------------
lc_run_id       = ${lc_run_id}
target_dir      = ${target_dir}
lc_result_dir   = ${lc_result_dir}
lm              = ${lm}
lm_vocab        = ${lm_vocab}
decoder         = ${decoder}
train_files     = ${train_files}
valid_files     = ${valid_files}
gpu             = ${gpu}
batch_size      = ${batch_size}
epochs          = ${epochs}
-----------------------------------------------------
" | tee ${lc_result_dir%/}/${lc_run_id}.log


for minutes in 1 10 100 1000
do
    run_id="${minutes}_min_${decoder}"

    echo "
    #################################################################################################
     Training on $minutes minutes, decoding=$decoder
     learning run id: $lc_run_id
     run id: $run_id
     target subdirectory: $lc_result_dir
    #################################################################################################
    "

    python3 run-train.py \
        --run_id ${run_id} \
        --target_dir ${lc_result_dir} \
        --minutes ${minutes} \
        --decoder ${decoder} \
        --lm ${lm} \
        --lm_vocab ${lm_vocab} \
        --gpu ${gpu} \
        --batch_size ${batch_size} \
        --epochs ${epochs} \
        --train_files ${train_files} \
        --valid_files ${valid_files} \
        2>&1 | tee ${lc_result_dir}/${run_id}.log # write to stdout and log file

    echo "
    #################################################################################################
     Finished $run_id
    #################################################################################################
    "
done
