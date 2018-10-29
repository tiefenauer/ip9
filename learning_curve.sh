#!/usr/bin/env bash
# set -xe
usage="$(basename "$0") [-h|--help] [-r|--run_id <string>] [-d|--destination <path>] [-t|--train_files <path>] [-v|--valid_files <path>] [-g|--gpu] [-b|--batch_size <int>] [-e|--epochs <int>] [-l|--lm <path>] [-a|--lm_vocab <path>] [-i|--language <string>] [-p|--dropouts] [-o|--optimizer <string>]
where:
    -h|--help                                show this help text
    -r|--run_id <string>                     run-id to use (used to resume training)
    -d|--destination <path>                  destination directory to store results
    -l|--lm <path>                           path to n-gram KenLM model (if possible binary)
    -a|--lm_vocab <path>                     path to file containing the vocabulary of the LM specified by -lm.
                                             The file must contain the words used for training delimited by space (no newlines)
    -i|--language <string>                   The language to train on ('en' or 'de'). This will determine the number of labels and therefore the model architecture. (default: en)
    -t|--train_files <path>                  one or more comma-separated paths to CSV files containing the samples to use for training
    -v|--valid_files <path>                  one or more comma-separated paths to CSV files containing the samples to use for validation
    -z|--test_files <path>                   one or more comma-separated paths to CSV files contatining the samples to use for testing
    -g|--gpu <int>                           GPU to use
    -b|--batch_size <int>                    batch size (default: 16)
    -e|--epochs <int>                        number of epochs to train (default: 30)
    -u|--valid_batches <int>                 number of batches to use for validation (default: all)
    -p|--dropouts                            whether to use dropout regularization (default: no)
    -x|--use_synth                           whether to use synthesized data for training (default: no)
    -o|--optimizer                           optimizer to use ('adam' or 'sgd'). (default: 'adam')

Create data to plot a learning curve by running a simplified version of the DeepSpeech-BRNN. This script will call run-train.py with increasing amounts of training data (1 to 1000 minutes).
For each amount of training data a separate training run is started. A unique run-id is assigned to each training run from which the value of each dimension can be derived.
"

# Defaults
lc_run_id="learning_run_$(uuidgen)"
lm=''
lm_vocab=''
language='en'
train_files=''
valid_files=''
test_files=''
target_dir=''
gpu=''
batch_size='16'
epochs='30'
valid_batches='0'
dropouts=''
use_synth=''
optimizer='sgd'

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
    -i|--language)
    language="$2"
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
    -z|--test_files)
    test_files="$2"
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
    -u|--valid_batches)
    valid_batches="$2"
    shift
    shift
    ;;
    -p|--dropout)
    dropouts='--dropouts'
    shift
    ;;
    -x|--use_synth)
    use_synth='--use_synth'
    shift
    ;;
    -o|--optimizer)
    optimizer="$2"
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
logfile=${lc_result_dir%/}/${lc_run_id}.log

SECONDS=0

git status > ${logfile}
git rev-parse HEAD >> ${logfile}
echo "
-----------------------------------------------------
 Start time: $(date)
 starting learning curve with the following parameters
-----------------------------------------------------
lc_run_id       = ${lc_run_id}
target_dir      = ${target_dir}
lc_result_dir   = ${lc_result_dir}
lm              = ${lm}
lm_vocab        = ${lm_vocab}
language        = ${language}
train_files     = ${train_files}
valid_files     = ${valid_files}
test_files      = ${test_files}
gpu             = ${gpu}
batch_size      = ${batch_size}
epochs          = ${epochs}
valid_batches   = ${valid_batches}
dropouts        = ${dropouts}
use_synth       = ${use_synth}
optimizer       = ${optimizer}
-----------------------------------------------------
" | tee -a ${logfile}

if [[ ${gpu} = '' ]]; then
    echo "Enter GPU # to use for training"
    read gpu
    echo "using GPU #${gpu} for all training runs!"
fi

cd ./src/

for minutes in 1 10 100 1000
do
    run_id="${minutes}_min"
    model_dir="${lc_result_dir}/${run_id}"

    echo "
    #################################################################################################
     Training on $minutes minutes
     learning run id: $lc_run_id
     run id: $run_id
     target subdirectory: $lc_result_dir
     model directory: $model_dir
    -------------------------------------------------------------------------------------------------
    "

    python3 run-train.py \
        --run_id ${run_id} \
        --target_dir ${lc_result_dir} \
        --minutes ${minutes} \
        --lm ${lm} \
        --lm_vocab ${lm_vocab} \
        --language ${language} \
        ${dropouts} \
        ${use_synth} \
        --optimizer ${optimizer} \
        --gpu ${gpu} \
        --batch_size ${batch_size} \
        --epochs ${epochs} \
        --valid_batches ${valid_batches} \
        --train_files ${train_files} \
        --valid_files ${valid_files} \
        2>&1 | tee ${lc_result_dir}/${run_id}.log # write to stdout and log file

    echo "
    -------------------------------------------------------------------------------------------------
     Testing model in $model_dir
     using samples from $test_files
    -------------------------------------------------------------------------------------------------
    "
    python3 run-test.py \
        --model_dir ${model_dir} \
        --test_files ${test_files} \
        --lm ${lm} \
        --lm_vocab ${lm_vocab} \
        --language ${language} \
        --gpu ${gpu}

    echo "
    -------------------------------------------------------------------------------------------------
     Finished training $run_id
     Model and test results saved in $model_dir
    #################################################################################################
    "
done

duration=$SECONDS

echo "
-----------------------------------------------------
 End time: $(date)
 Finished in ${duration} seconds
-----------------------------------------------------
"