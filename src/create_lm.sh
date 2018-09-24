#!/usr/bin/env bash
# create n-gram Language Model on ~2.2M German Wikipedia articles using with KenLM.
# The articles will be downloaded from the Wikipedia dump, which requires ~5 GB of free storage.
# The articles ar preprocessed by removing XML tags and quotes. Preprocessing requires uncompressing and will
# take ~3-4 hours and ~xxx of free storage.
# For a some statistics of Wikipedias see https://meta.wikimedia.org/wiki/List_of_Wikipedias
#
# Note: There is a blog post about creating a LM for English using the Bible. However, this blog post uses bzcat to stream
# the content of the compressed file and pipe it through a Python script for preprocessing. However, this is not a very
# performant way to train an LM since lmplz will complain with something like "File stdin isn't normal. Using slower read()..."
# To maximize performance, this scripts writes the preprocessed data is written to disk and read from there.
#
# Positional arguments:
#   N: defines the order of the n-gram model. Default=2
# Flags:
#  -r or --remove_files: Whether to remove artifacts after processing. Default=false. NOTE: If set to true the script will
#                        need to download and preprocess the files again if a model of different order should be trained
#                        on the same data! Downloading and preprocessing require approximately xxx hours!
#set -xe

# replace this with the path to your KenLM binary folder
# see https://kheafield.com/code/kenlm/ for details
KENLM_BIN_PATH=/home/daniel/kenlm/build/bin

N=${1:-2}
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

corpus_name="wiki_de"

# #################################
# Paths and filenames
# #################################
download_url="http://download.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2"
download_dir="../lm_data/${corpus_name}"  # directory for wikipedia dump and processed articles
target_file=$download_dir/$(basename $download_url)  # get corpus file name from url and corpus name
cleaned_dir="${download_dir}/cleaned" # directory for cleaned articles (conmpressed)
dewiki_raw="${download_dir}/${corpus_name}.raw" # file containing all articles (uncompressed)
dewiki_processed="${download_dir}/${corpus_name}.txt" # file containing all preprocessed articles (uncompressed)

lm_dir="../lm/${corpus_name}" # base directory for trained LM
lm_basename="${corpus_name}_${N}_gram"
lm_arpa="${lm_dir}/${lm_basename}.arpa" # ARPA file
lm_binary="${lm_dir}/${lm_basename}.klm" # binary file for faster loading (see https://kheafield.com/code/kenlm/structures/)
lm_vocab="${lm_dir}/${lm_basename}.words" # ARPA file

mkdir -p $download_dir
mkdir -p $lm_dir
# #################################

echo "creating $N-gram model from German Wikipedia"

if [ ! -f $target_file ]; then
    echo "downloading corpus ${corpus_name} from download_url and saving in $target_file"
    echo "This can take up to an hour. Get a coffee or something..."
    wget -O $target_file $download_url
fi

if [ ! -d "$cleaned_dir" ]; then
    echo "extracting/cleaning text from Wikipedia data base dump at $target_file and saving to $cleaned_dir"
    echo "This can take several hours. Go to sleep or something..."
    mkdir -p $cleaned_dir
    pv < $(python2 ./lm/WikiExtractor.py -c -b 25M -o $cleaned_dir $target_file)
fi

#find $cleaned_dir -name '*bz2' |\! -exec bzip2 -k -c -d {} \; | \

echo "uncompressing and preprocessing cleaned articles from $cleaned_dir and writing to $dewiki_processed"
pv -cN source < $(find $cleaned_dir -name '*bz2' -exec bzcat {} + \
     | tee >(   sed 's/<[^>]*>//g' \
              | sed 's|["'\''„“‚‘]||g' \
              | python3 ./lm/create_lm.py $lm_vocab > $dewiki_processed \
          ))

#echo "Number of articles: "
#grep -o "<doc" $dewiki_raw | wc -w

#echo "removing XML tags and quotations in $dewiki_raw"
#sed -i 's/<[^>]*>//g' $dewiki_raw
#sed -i 's|["'\''„“‚‘]||g' $dewiki_raw

#echo "preprocessing text by tokenizing words and removing punctuation"
#cat $dewiki_raw | python3 ./lm/create_lm.py $lm_vocab > $dewiki_processed
#
#echo "Training $N-gram KenLM model with data from $dewiki_raw and saving ARPA file to $lm_arpa"
##echo "#newlines  #words  #bytes"
#cat $dewiki_raw | python3 ./lm/create_lm.py | $KENLM_BIN_PATH/lmplz -o $N -S 40% <$dewiki_processed >$lm_arpa
#
#echo "Building binary file from $lm_arpa and saving to $lm_binary"
#$KENLM_BIN_PATH/build_binary $lm_arpa $lm_binary
#
#echo "Checking newly built model: Please enter a sentence in the language the model was trained on: "
#read test_sentence
#python3 -c "from lm.create_lm import *; check_lm('$lm_binary', 'vocab.txt', '$test_sentence')"

if $REMOVE_FILES; then
    echo "removing downloaded file: ${}"
    rm -rf $target_file
    echo "removing extracted files: ${cleaned_dir}"
    rm -rf $cleaned_dir
    rm -rf $dewiki_raw
    rm -rf $dewiki_processed
fi

#echo $extract_cmd
#echo "#newlines  #words  #bytes"
#extract_cmd | python3 lines_to_normalized_words.py | wc
#$extract_cmd | python3 ./lm/create_lm.py