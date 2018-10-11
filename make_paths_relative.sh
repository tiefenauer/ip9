# processes a standard LibriSpeech corpus file by removing the root path (i.e. make the audio file paths relative to the root path) and adding length information
# usage: ./make_path_relative.sh /path/to/common-voice/corpus/cv-valid-train.csv > /path/to/output/dir/cv-valid-train-rel.csv
echo 'wav_filename,wav_filesize,wav_length,transcript'
root_path='\/media\/D1\/daniel.tiefenauer\/corpora\/cv\/'
while IFS=, read -r wav_filename wav_filesize transcript
do
	if [[ ${wav_filename} = 'wav_filename' ]]; then
		continue
	fi
        wav_length=$(soxi -D $wav_filename)
        wav_path=${wav_filename//$root_path/}

	echo "${wav_path},${wav_filesize},${wav_length},${transcript}"
done < $1	
