#!/usr/bin/env bash
# create n-gram Language Model on ~2.2M German Wikipedia articles using with KenLM.
# For a some statistics of Wikipedias see https://meta.wikimedia.org/wiki/List_of_Wikipedias
# The articles will be downloaded as Wikipedia dump, (~5-6 GB) and preprocessed to plaintext.
# The whole process will take several hours and requires ~25GB free storage.
#
# This script will perform the following steps (created artifacts in brackets)
# 1) download the dump file ({language}wiki-latest-pages-articles.xml.bz2)
# 2) convert the articles to raw corpus file (wiki_{language}.txt)
# 4) create a vocabulary for the unique words in the dump (wiki_{language}.words)
# 5) train a n-gram KenLM model (wiki_{language}_n_gram.arpa and wiki_{language}_n_gram.klm, where n denotes the order of the LM)
#
# Note 1: to convert a Wikipedia dump to text there is a tool called Wiki Parser (https://dizzylogic.com/wiki-parser/)
# However this tool only runs in windows and cannot be used in this script. You can however use this script to generate
# a text file (articles_in_plain_text.txt) and place it in ./lm_data/wiki_{language}. The script will then use this file as base
# to create an input file for KenLM (KenLM expects the input file to contain one sentence per line, ending with a dot).
# If there is no such file, the Wikipedia dump is processed using Wikiextractor (https://github.com/attardi/wikiextractor).
# However, this requires decompressing the dump
#
# Note 2: There is a blog post about creating a KenLM for English using the Bible (http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html).
# However, this blog post uses bzcat to stream the content of the compressed file and pipe it through a Python script
# for preprocessing. However, this is not a very performant way to train an LM since lmplz will complain with something
# like "File stdin isn't normal. Using slower read()...". To maximize performance, this scripts writes the preprocessed
# data is written to disk and read from there.
#
# Positional arguments:
#   LANGUAGE: defines the language ('en', 'fr', 'de', ...). Default='de')
#   N: defines the order of the n-gram model. Default=2
# Flags:
#  -r or --remove_files: Whether to remove artifacts after processing. Default=false. NOTE: If set to true the script will
#                        need to download and preprocess the files again if a model of different order should be trained
#                        on the same data! Downloading and preprocessing require approximately xxx hours!
#
# example usage (create a 4-gram model for German)
# ./create_lm.sh de 4
# set -xe

# first argument defines the language and defaults to German
LANGUAGE=${1:-"de"}
# second argument defines the order of the LM and defaults to 2 (2-gram model)
N=${2:-2}
REMOVE_FILES=false

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -r|--remove_files) # remove downloaded/extracted files after processing
    remove_files=true
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# replace this with the path to your KenLM binary folder
# see https://kheafield.com/code/kenlm/ for details
KENLM_BIN_PATH=/home/daniel/kenlm/build/bin

# set this to a directory with at least 40GB free storage
tmp_dir=/home/daniel/tmp

corpus_name="wiki_${LANGUAGE}"

# #################################
# Paths and filenames
# #################################
download_url="http://download.wikimedia.org/${LANGUAGE}wiki/latest/${LANGUAGE}wiki-latest-pages-articles.xml.bz2"
download_dir="../lm_data/${corpus_name}"  # directory for wikipedia dump and processed articles
target_file=$download_dir/$(basename $download_url)  # get corpus file name from url and corpus name
wikiparser_result=${download_dir}/articles_in_plain_text.txt # (optional) result from WikiParser
cleaned_dir="${download_dir}/${corpus_name}_clean" # directory for cleaned articles (conmpressed)
corpus_file="${download_dir}/${corpus_name}.txt" # file containing all preprocessed articles (uncompressed)

lm_dir="../lm/${corpus_name}" # base directory for trained LM
lm_vocab="${lm_dir}/${corpus_name}.words" # Vocabulary (same for all order LMs, therefore order is not in file name)
lm_basename="${corpus_name}_${N}_gram"
lm_arpa="${lm_dir}/${lm_basename}.arpa" # ARPA file
lm_binary="${lm_dir}/${lm_basename}.klm" # binary file for faster loading (see https://kheafield.com/code/kenlm/structures/)

# create target directories if the don't exist yet
mkdir -p $download_dir
mkdir -p $lm_dir

# put the source folder on PYTHONPATH
export PYTHONPATH=./src:$PYTHONPATH
# #################################

echo "creating $N-gram model from German Wikipedia dump"
echo "time indications are based upon personal experience when training on my personal laptop (i7, 4 cores, 8GB RAM, SSD)"

# set flag to recreate vocab: flag is stored as variable because it must be assignable
# recreate vocab if vocab file does not exist, but even if it exists it must be recreated when $corpus_file is recreated
[ ! -f $lm_vocab ]
recreate_vocab=$?

if [ ! -f $target_file ]; then
    echo "downloading corpus ${corpus_name} from download_url and saving in $target_file"
    echo "This can take up to an hour (Wiki servers are slow). Have lunch or something..."
    wget -O $target_file $download_url
fi

if [ -f $wikiparser_result ] && [ ! -f "${corpus_file}.bz2" ]; then
    echo "found Wiki Parser result file at $wikiparser_result. Using this file."
    num_lines=$(wc -l $wikiparser_result)
    python3 ./lm/corpus2kenlm.py --input $wikiparser_result --output $corpus_file --num_lines $num_lines
    recreate_vocab = 1

elif [ ! -f "${corpus_file}.bz2" ] ; then
    if [ ! -d $cleaned_dir ] ; then
        echo "Extracting/cleaning text from Wikipedia data base dump at $target_file using WikiExtractor."
        echo "Cleaned articles are saved to $cleaned_dir"
        echo "This will take 2-3 hours. Have a walk or something..."
        mkdir -p $cleaned_dir
        python2 ./lm/WikiExtractor.py -c -b 25M -o $cleaned_dir $target_file
    fi
    echo "Uncompressing and preprocessing cleaned articles from $cleaned_dir"
    echo "All articles will be written to $corpus_file (1 sentence per line, without dot at the end)."
    echo "All XML tags will be removed. Numeric word tokens will be replaced by the <unk> token."
    echo "Non-ASCII characters will be replaced with their closest ASCII equivalent (if possible), but umlauts will be preserved!"
    echo "This will take some time (~4h). Go to sleep or something..."
    result=$(find $cleaned_dir -name '*bz2' -exec bzcat {} \+ \
            | pv \
            | tee >(    sed 's/<[^>]*>//g' \
                      | sed 's|["'\''„“‚‘]||g' \
                      | python3 ./lm/create_corpus.py ${LANGUAGE} > ${corpus_file} \
                   ) \
            | grep -e "<doc" \
            | wc -l)
    echo "Processed ${result} articles and saved raw text in $corpus_file"
    recreate_vocab = true
fi

if ((${recreate_vocab} == true)) ; then
    echo "(re-)creating vocabulary of $corpus_file and saving it in $lm_vocab. "
    echo "This usually takes around half an hour. Get a coffee or something..."
    grep -oE '\w+' ${corpus_file} | pv -s $(stat --printf="%s" ${corpus_file}) | sort -u -f > ${lm_vocab}
fi

if [ -f ${corpus_file} ]; then
    echo "Processed $(cat ${corpus_file} | wc -l) sentences"
    echo "Processed $(cat ${corpus_file} | wc -w) words"
    echo "Processed $(cat ${lm_vocab} | wc -l) unique words"

    echo "compressing $corpus_file. File size before:"
    du -h ${corpus_file}
    bzip2 ${corpus_file}
    echo "done! Compressed file size:"
    du -h ${corpus_file}.bz2
fi


if [ ! -f $lm_arpa ]; then
    echo "Training $N-gram KenLM model with data from $corpus_file.bz2 and saving ARPA file to $lm_arpa"
    echo "This can take several hours, depending on the order of the model"
    ${KENLM_BIN_PATH}/lmplz -o $N -T $tmp_dir --skip_symbols -S 40% <${corpus_file}.bz2 \
    | ${KENLM_BIN_PATH}/build_binary /dev/stdin ${lm_binary}
fi

if [ ! -f $lm_binary ]; then
    echo "Building binary file from $lm_arpa and saving to $lm_binary"
    echo "This should usually not take too much time even for high-order models"
    ${KENLM_BIN_PATH}/build_binary ${lm_arpa} ${lm_binary}
fi

if ${REMOVE_FILES}; then
    echo "removing downloaded file: ${target_file}"
    rm -rf ${target_file}
    echo "removing extracted files: ${cleaned_dir}"
    rm -rf ${cleaned_dir}
    rm -rf ${corpus_file}
fi