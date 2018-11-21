#!/usr/bin/env bash
# set -xe
usage="$(basename "$0") [-h|--help] [-o|--order <int>] [-l|--language {'en'|'de'|'fr'|'it'|...}] [-d|--data_structure {'probing'|'trie'}] [-t|--target_dir <string> ] [-r|--remove_artifacts remove_artifacts]

Create n-gram Language Model on ~2.2M Wikipedia articles using KenLM.
Parameters:
    -h|--help              show this help text
    -o|--order             set the order of the model, i.e. the n in n-gram (default: 4)
    -l|--language          ISO 639-1 code of the language to train on (default: de)
    -d|--data_structure    data structure to use (use 'trie' or 'probing'). See https://kheafield.com/code/kenlm/structures/ for details. (default: trie)
    -t|--target_dir        target directory to write to
    -r|--remove_artifacts  remove intermediate artifacts after training. Only set this flag if you really don't want to train another model because creating intermediate artifacts can take a long time. (default: false)

EXAMPLE USAGE: create a 5-gram model for German using the 40k most frequent words from the Wikipedia articles, using probing as data structure and removing everything but the trained model afterwards:

./create_lm.sh -l de -o 5 -r

Make sure the target directory specified by -t has enough free space (around 20-30G). KenLM binaries (lmplz and build_binary) need to be on the path. See https://kheafield.com/code/kenlm/ on how to build those.

The following intermediate artifacts are created and may be removed after training by setting the -r flag:
- {target_dir}/tmp/[language]wiki-latest-pages-articles.xml.bz2: Downloaded wikipedia dump
- {target_dir}/tmp/[language]_clean: directory containing preprocessed Wikipedia articles
- {target_dir}/tmp/wiki_[language].txt.bz2: compressed file containing the Wikipedia corpus used to train the LM (raw text contents of the Wikipedia articles one sentence per line)
- {target_dir}/tmp/wiki_[language].counts: file containing the full vocabulary of the corpus and the number of occurrences of each word (sorted descending by number of occurrences)
- {target_dir}/tmp/wiki_[language].vocab: file containing the most frequent words of the corpus used for training (as defined by the -t argument) in the format expected by KenLM (words separated by spaces)
- {target_dir}/tmp/wiki_[language].arpa: ARPA file used to create the KenLM binary model

The following result files are created and will not be removed:
- {target_dir}/wiki_[language].klm: final KenLM n-gram LM in binary format.
"

# Defaults
order=4
language='de'
data_structure=trie
target_dir='./lm'
remove_artifacts=false

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
    -o|--order)
    order="$2"
    shift
    shift
    ;;
    -l|--language)
    language="$2"
    shift
    shift
    ;;
    -d|--data_structure)
    data_structure="$2"
    shift
    shift
    ;;
    -t|--target_dir)
    target_dir="$2"
    shift
    shift
    ;;
    -r|--remove_artifacts)
    remove_artifacts=true
    shift
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# #################################
# Paths and filenames
# #################################
corpus_name="wiki_${language}"
lm_basename="${corpus_name}_${order}_gram"

target_dir="${target_dir}/${corpus_name}" # directory to store model
tmp_dir="${target_dir}/tmp"  # directory for intermediate artifacts

cleaned_dir="${tmp_dir}/${corpus_name}_clean" # directory for WikiExtractor
corpus_file="${tmp_dir}/${corpus_name}.txt" # uncompressed corpus
lm_counts="${tmp_dir}/${corpus_name}.counts" # corpus vocabulary with counts (all words)
lm_arpa="${tmp_dir}/${lm_basename}.arpa" # ARPA file

lm_binary="${target_dir}/${lm_basename}.klm" # KenLM binary file (this is the result of the script)

# create target directories if the don't exist yet
mkdir -p ${target_dir}
mkdir -p ${tmp_dir}
# #################################

echo "creating $order-gram model from Wikipedia dump"
echo "time indications are based upon personal experience when training on my personal laptop (i7, 4 cores, 8GB RAM, SSD)"

# #################################
# STEP 1: Download the Wikipedia dump in the given language if necessary
# For a some statistics of Wikipedias see https://meta.wikimedia.org/wiki/List_of_Wikipedias
# #################################
download_url="http://download.wikimedia.org/${language}wiki/latest/${language}wiki-latest-pages-articles.xml.bz2"
target_file=${tmp_dir}/$(basename ${download_url})  # get corpus file name from url and corpus name
if [[ ! -f ${target_file} ]]; then
    echo "downloading corpus ${corpus_name} from ${download_url} and saving in ${target_file}"
    echo "This can take up to an hour (Wiki servers are slow). Have lunch or something..."
    wget -O ${target_file} ${download_url}
fi

# #################################
# STEP 2: Create corpus from dump if necessary
# Use WikExtractor (see https://github.com/attardi/wikiextractor for details)
# #################################
if [[ ! -f "${corpus_file}" ]] ; then
    cd ./src/
    if [[ ! -d ${cleaned_dir} ]] ; then
        echo "Extracting/cleaning text from Wikipedia data base dump at ${target_file} using WikiExtractor."
        echo "Cleaned articles are saved to ${cleaned_dir}"
        echo "This will take 2-3 hours. Have a walk or something..."
        mkdir -p ${cleaned_dir}
        python3 ./lm/WikiExtractor.py -c -b 25M -o ${cleaned_dir} ${target_file}
    fi
    echo "Uncompressing and preprocessing cleaned articles from $cleaned_dir"
    echo "All articles will be written to $corpus_file (1 sentence per line, without dot at the end)."
    echo "All XML tags will be removed. Numeric word tokens will be replaced by the <num> token."
    echo "Non-ASCII characters will be replaced with their closest ASCII equivalent (if possible), but umlauts will be preserved!"
    echo "This will take some time (~4h). Go to sleep or something..."
    export PYTHONPATH=$(pwd)
    result=$(find ${cleaned_dir} -name '*bz2' -exec bzcat {} \+ \
            | pv \
            | tee >(    sed 's/<[^>]*>//g' \
                      | sed 's|["'\''„“‚‘]||g' \
                      | python3 ./lm/create_corpus.py ${language} > ${corpus_file} \
                   ) \
            | grep -e "<doc" \
            | wc -l)
    echo "Processed ${result} articles and saved raw text in $corpus_file"

    echo "Processed $(cat ${corpus_file} | wc -l) sentences"
    echo "Processed $(cat ${corpus_file} | wc -w) words"
    echo "Processed $(cat ${corpus_file} | xargs -n1 | sort | uniq -c) unique words"

    echo "(re-)creating vocabulary of $corpus_file because corpus file has changed"
    echo "This usually takes around half an hour. Get a coffee or something..."
    ./create_corpus_vocab.sh ${corpus_file} --target_dir ${target_dir}
fi

echo "compressing $corpus_file. File size before:"
du -h ${corpus_file}
bzip2 -k ${corpus_file}
echo "done! Compressed file size:"
du -h ${corpus_file}.bz2

if [[ ! -f ${lm_arpa} ]]; then
    echo "Training $order-gram KenLM model with data from $corpus_file.bz2 and saving ARPA file to $lm_arpa"
    echo "This can take several hours, depending on the order of the model"
    lmplz --order ${order} \
          --temp_prefix ${tmp_dir} \
          --memory 40% \
          --arpa ${lm_arpa} \
          --prune 0 0 0 1 <${corpus_file}.bz2
fi

if [[ ! -f ${lm_binary} ]]; then
    echo "Building binary file from $lm_arpa and saving to $lm_binary"
    echo "This should usually not take too much time even for high-order models"
    build_binary -a 255 \
                 -q  8 \
                 ${data_structure} \
                 ${lm_arpa} \
                 ${lm_binary}
fi

if ${remove_artifacts}; then
    echo "removing intermediate artifacts in ${tmp_dir}"
    rm -rf ${tmp_dir}
fi
