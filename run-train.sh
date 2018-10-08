#!/usr/bin/env bash
# set -xe
usage="$(basename "$0") [-h|--help] [-r|--run_id <string>] [-d|--destination <path>] [-x|--decoder <string>] [-l|--lm <path>] [-a|--lm_vocab <path>] [-t|--train_files <path>] [-v|--valid_files <path>] [-m|--minutes <int>] [-g|--gpu <int>] [-b|--batch_size <int>] [-e|--epochs <int>]
where:
    -h|--help                                show this help text
    -r|--run_id <string>                     run-id to use (used to resume training)
    -d|--destination <path>                  destination directory to store results
    -x|--decoder <'beamsearch'|'bestpath'>   decoder to use (default: beamsearch)
    -l|--lm                                  path to n-gram KenLM model (if possible binary)
    -a|--lm_vocab                            path to file containing the vocabulary of the LM specified by -lm. The file must contain the words used for training delimited by space (no newlines)
    -t|--train_files <path>                  one or more comma-separated paths to CSV files containing the corpus files to use for training
    -v|--valid_files <path>                  one or more comma-separated paths to CSV files containing the corpus files to use for validation
    -m|--minutes <int>                       number of minutes of audio for training. If not set or set to 0, all training data will be used (default: 0)
    -g|--gpu <int>                           GPU to use (default: 2)
    -b|--batch_size <int>                    batch size
    -e|--epochs <int>                        number of epochs to train

Train a simplified model of the DeepSpeech RNN on a given corpus of training- and validation-data.
"

# Defaults
run_id=''
target_dir='/home/daniel_tiefenauer'
minutes=0
gpu='2'
batch_size='16'
epochs='20'
decoder='beamsearch'
lm='./lm/libri-timit-lm.klm'
lm_vocab='./lm/words.txt'
train_files='/media/D1/readylingua-en/readylingua-en-train.csv'
valid_files='/media/D1/readylingua-en/readylingua-en-dev.csv'

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
    echo ${usage}
    shift # past argument
    exit
    ;;
    -r|--run_id)
    run_id="$2"
    shift # past argument
    shift # past value
    ;;
    -d|--destination)
    target_dir="$2"
    shift # past argument
    shift # past value
    ;;
    -x|--decoder)
    decoder="$2"
    shift # past argument
    shift # past value
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
    shift # past argument
    shift # past value
    ;;
    -v|--valid_files)
    valid_files="$2"
    shift # past argument
    shift # past value
    ;;
    -m|--minutes)
    minutes="$2"
    shift # past argument
    shift # past value
    ;;
    -g|--gpu)
    gpu="$2"
    shift # past argument
    shift # past value
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
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo ' '
echo '-----------------------------------------------------'
echo ' starting training run with the following parameters'
echo '-----------------------------------------------------'
echo run_id       = "${run_id}"
echo target_dir   = "${target_dir}"
echo minutes      = "${minutes}"
echo decoder      = "${decoder}"
echo lm           = "${lm}"
echo lm_vocab     = "${lm_vocab}"
echo train_files  = "${train_files}"
echo valid_files  = "${valid_files}"
echo gpu          = "${gpu}"
echo batch_size   = "${batch_size}"
echo epochs       = "${epochs}"
echo '-----------------------------------------------------'
echo ' '

cd ./src/

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
