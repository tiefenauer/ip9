#!/usr/bin/env bash
# Download LibriSpeech raw text and create vocabulary of n most frequent words

n=250000 # use 40k most frequent words

# download file
wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz

# decompress file
gunzip librispeech-lm-norm.txt.gz

# count word occurrences and keep n most frequent words
cat librispeech-lm-norm.txt |
    pv -s $(stat --printf="%s" librispeech-lm-norm.txt) | # show a progress bar
    tr '[:upper:]' '[:lower:]' | # lowercase everything
    tr -s '[:space:]' '\n' | # replace spaces with one newline
    sort | # sort alphabetically
    uniq -c | # count occurrences
    sort -bnr | # numeric sort
    tr -d '[:digit:] ' | # remove counts from lines
    head -${n} | # keep n most frequent words words
    tr '\n' ' ' > lm.vocab # replace line breaks with spaces and write to lm.vocab