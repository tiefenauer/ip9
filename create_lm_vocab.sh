#!/usr/bin/env bash
usage="$(basename "$0") <string> [-h|--help] [-f|--corpus_file <string>]

Create vocabulary with 40k/80k/120k most frequent words from corpus file.

Positional parameters:
    absolute path to corpus file containing normalized text

Named parameters
    -h|--help              show this help text
    -t|--target_dir       absolute path to target directory to save vocabulary
"

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
    -t|--target_dir)
    target_dir="$2"
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

corpus_file=$1
corpus_filename=$(basename -- ${corpus_file})
corpus_filename="${corpus_filename%.*}"
vocab_counts=${corpus_file%.*}.counts

if [[ ! -f "${vocab_counts}" ]] ; then
    echo "counting word occurrences and saving them in $vocab_counts..."
    cat ${corpus_file} |
        pv -s $(stat --printf="%s" ${corpus_file}) | # show progress bar
        tr '[:upper:]' '[:lower:]' | # make everything lowercase
        tr -s '[:space:]' '\n' | # replace any number of spaces with one newline (one word per line)
        grep -v '^\s*$' | # remove empty lines
        grep -Ev '[0-9]' | # remove words containing numbers
#        awk 'length($0)>1' | # remove words with length 1
        sort | uniq -c | sort -bnr > ${vocab_counts} # sort alphanumeric, count unique words, then sort numeric
    echo '...done!'
fi

total_sum=$(echo $(cat ${vocab_counts} |
          tr -sc '[:digit:]' '+' |
          sed 's/+$//') |
          bc) # sum everything up
echo "total number of words in vocabulary: $total_sum"

for top_words in 40 80 160
do
    vocab_file=${target_dir}/${corpus_filename}_${top_words}k.vocab
    n=$((${top_words}*1000))
    echo "writing $n top words to vocabulary"
    head -${n} ${vocab_counts} |
        tr -d '[:digit:] ' | # remove counts from lines
        tr '\n' ' ' > ${vocab_file} # replace newline with spaces (expected input format for KenLM)

    top_sum=$(echo $(head  -${n} ${vocab_counts} |
              tr -sc '[:digit:]' '+' | #remove everything non-numeric by a plus sign
              sed 's/+$//') | # remove last plus sign
              bc) # sum everything up
    echo "number of words in vocabulary: $top_sum"

    fraction=$(echo "scale=2 ; 100 * $top_sum / $total_sum" | bc)
    echo "Top $n words make up $fraction% of words"
done