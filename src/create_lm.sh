#!/usr/bin/env bash
# create n-gram Language Model with KenLM
# Positional arguments:
#   CORPUS_NAME: must be one of bible_en | bible_de | novels_de
#   N: defines the n in n-gram (default: 2)
#set -xe

# replace this with the path to your KenLM binary folder
# see https://kheafield.com/code/kenlm/ for details
KENLM_BIN=/home/daniel/kenlm/build/bin

CORPUS_NAME=$1
N=${2:-2}

echo "creating $N-gram model for corpus $CORPUS_NAME"

case $CORPUS_NAME in
"bible_en")
    corpus_url="https://github.com/vchahun/notes/raw/data/bible/bible.en.txt.bz2"
    corpus_file="bible.en.txt.bz2"
    extract_text="bzcat $corpus_file"
    ;;
"bible_de")
    corpus_url="http://www.wordproaudio.com/bibles/resources/download/zip/de.zip"
    corpus_file="bible_de.zip"
    extract_text="unzip -c $corpus_file *.htm"
    ;;
"novels_de")
    corpus_url="https://ndownloader.figshare.com/files/3686778"
    corpus_file="novels.de.txt.zip"
    extract_text="unzip -c $corpus_file DE_*.txt"
    ;;
*)
    echo "ERROR: corpus name must be set"
    exit 1
esac

if [ ! -f $corpus_file ]; then
    echo "downloading corpus $CORPUS_NAME from $corpus_url and saving in $corpus_file"
    wget -O $corpus_file $corpus_url
fi

echo "newlines  words  bytes"
$extract_text | python3 create_lm.py | wc

echo "training LM"
$extract_text | python3 create_lm.py | $KENLM_BIN/lmplz -o 2 -S 20% > $CORPUS_NAME.arpa

echo "building binary"
$KENLM_BIN/build_binary $CORPUS_NAME.arpa $CORPUS_NAME.klm

echo "checking newly built model"
test_sentence='Language modeling is fun .'
python3 -c "from create_lm import *; check_lm('$CORPUS_NAME.klm', 'vocab.txt', '$test_sentence')"