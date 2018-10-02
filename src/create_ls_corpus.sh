#!/usr/bin/env bash

librispeech_mp3_url="http://www.openslr.org/resources/12/original-mp3.tar.gz"
librispeech_books_url="http://www.openslr.org/resources/12/original-books.tar.gz"

tmp_dir="./tmp"
mkdir -p ${tmp_dir}

librispeech_mp3_tar="${tmp_dir}/original-mp3.tar.gz"
librispeech_mp3_target="${tmp_dir}/original-mp3"
mkdir -p ${librispeech_mp3_target}

if [ ! -f ${librispeech_mp3_target} ]; then
    echo "Downloading LibriSpeech data from $librispeech_mp3_url and extracting to $librispeech_mp3_target"
    wget -O ${librispeech_mp3_tar} ${librispeech_mp3_url}
    tar -xf ${librispeech_mp3_tar} -C ${librispeech_mp3_target}
fi

rm -rf ${tmp_dir}