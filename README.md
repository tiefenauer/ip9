# IP9
Code for my master thesis at FHNW. The thesis suggests a pipeline with the following stages to perform [_Forced Alignment_](http://linguistics.berkeley.edu/plab/guestwiki/index.php?title=Forced_alignment) for a combination of audio and text. The pipeline has the following stages:

* **Preprocessing**: normalize audio and text
* **Voice Activity Detection (VAD)**: split the audio into voiced segments
* **Automatic Speech Recognition (ASR)**: transcribe each voiced segment
* **Global Sequence Alignment (GSA)**: align each transcript with the text 

The main idea behind this is that the GSA stage only needs very low-quality transcripts in order to work. Such transcripts can be obtained from a simple ASR system that only requires very little training data (compared to an ASR system trained for speech recognition).  

This repository contains everything needed to build and run the _Forced Alignment_ pipeline. See [the project report](./doc/p9_tiefenauer.pdf) for details. 

## Setup
The code was created to run on Linux (18.04 LTS). Other platforms like Windows are not supported. Some of the code might run on those platforms, but this can not be guaranteed. Some dependencies like [DeepSpeech](https://github.com/mozilla/DeepSpeech) are known to work only on Unix based systems. 

### Dependencies

* This application uses [Pydub](http://pydub.com/) which means you will need **libav** or **ffmpeg** on your `PATH`. See the [Pydub Repository](https://github.com/jiaaro/pydub#installation) for further instructions.
* Visual C++ build tools to work with [webrtcvad](https://github.com/wiseman/py-webrtcvad) (google it for download link): must be installed before installing the python requirements (see below)!
* [Sox](http://sox.sourceforge.net/) is used to convert MP3 into PCM-WAV and must be on the `PATH`. You also need to install the handler for MP3 files. On Linux it this is easiest done by executing the following commants:
  * `sudo apt-get install sox`: to install Sox
  * `sudo apt-get install libsox-fmt-all`: to install all file handlers 
* [PipeViewer](http://www.ivarch.com/programs/pv.shtml) for some bash scripts
* [pygit2](https://www.pygit2.org) is used to include the commit ID when plotting results. It uses [libgit2](https://libgit2.org/), which must be installed before installing the pygit2 Python package. See [the installation instructions](https://www.pygit2.org/install.html#quick-install) on how to quickly install libgit2.
* [KenLM](https://github.com/kpu/kenlm) is used to create an n-gram Language Model. It is assumed that `lmplz` and `build_binary` are on `$PATH`. See [the KenLM docs](https://kheafield.com/code/kenlm/) for more information about how to build those binaries from source.

### Installation
1. Clone [the repository](https://github.com/tiefenauer/ip9): `git clone git@github.com:tiefenauer/ip9.git` 
2. Install Python requirements: `pip install -r requirements.txt`. **IMPORTANT: Make sure that Keras 2.2.2 is installed! (`pip show keras`)**
3. Install [TensorFlow](https://www.tensorflow.org/install/): TF is not included in `requirements.txt` because you can choose between the `tensorflow` (no GPU acceleration) and `tensorflow-gpu` (with GPU-acceleration). If your computer does not have a CUDA-supported GPU (like mine does) you will install the former, else the latter. Installing `tensorflow-gpu` on a computer without GPU does not work.

## Training the ASR stage

The pipeline approach uses a simplified version of _DeepSpeech_.

### Create corpora from raw data

To train the Neural Network for ASR, corpora must be created from raw data. The following scripts in the main folder process this raw data and store the result in a target directory. Type `python {script name} -h` to see how to use the files.

| corpus | script to create corpus | estimated time needed |
|---|---|---|
| LibriSpeech | [`create_ls_corpus.py`](./src/create_ls_corpus.py) | 6-7h |
| ReadyLingua | [`create_rl_corpus.py`](./src/create_rl_corpus.py) | 2-3m |

The scripts will read the raw data (audio, transcripts) and normalize them. All audio data will be converted to a 16kHz RAW-WAV (PCM) using 16-bit signed Integer encoding (little endian). The transcript of each segment is normalized by lowercasing, punctuation removal, unidecoding, removal of whitespace. The full transcript will be stored as-is (no normalization). The script will extract the metadata (segmentation information, transcripts, etc.) a file called `index.csv` in the target directory. This file has the following structure:

```csv
id,entry_id,subset,language,audio_file,start_frame,end_frame,duration,transcript,numeric
```

whereas the keys have the following meaning:

* `id`: ID of the voiced segment
* `entry_id`: ID of the corpus entry (audio/transcript combination) the segment appears in
* `subset`: to which set a segment belons (`train`, `dev` or `test`)
* `language`: language of the audio/transcript
* `audio_file`: path (relative to `index.csv`) to the normalized audio file
* `start_frame`: start frame of the segment within `audio_file`
* `end_frame`: end frame of the segment within `audio_file`
* `duration`: length of the segment in seconds
* `transcript`: normalized transcript of the segment

### DeepSpeech

The script [`run-train.py`](./src/run-train.py) trains the simplified model on a training set. The script [`run-test.py`](./src/run-test.py) runs inference on a test set. Both scripts rely on the training-, validation- and test-set being available in the expected format. The expected format just a directory with one PCM-16 encoded `*.wav` file for each training sample. The transcript of each training sample is saved to CSV files together with additional information. Since the samples are split into training-, validation- and test-set, samples in each set are defined by a separate csv files. Each CSV file has the following colums:

```csv
wav_filename,wav_filesize,wav_length,transcript
```

whereas the keys have the following meaning:

- `wav_filename`: Path to the wav file. Note that the path must be absolute, which means the corpus is not easily portable.
* `wav_filesize`: size of the wav file in bytes
* `wav_length`: length of the recording in seconds
* `transcript`: arbitrary string containing the start frame, end frame and normalized transcript for the recording. The string must be lowercase and only contain characters from the alphabet the model is trained on. 

**Example CSV file for corpus**:
```csv
wav_filename,wav_filesize,wav_length,transcript
/path/to/001.wav,179882,5.6199375,some transcript
/path/to/002.wav,123402,3.8549375,some other transcript
...
```

The set of CSV files and accompanying audio files constitute a corpus that can be used to train the simplified model. The script [`corpus2csv.py`](./src/corpus2csv.py) can by used to create such corpus from the corpus described above. This script can also synthesize data while creating the corpus. Type `python corpus2csv.py -h` to see how to use it.

## Evaluating the pipeline

After having trained a simplified model, it can be used for inference in the pipeline. Its outputs are then aligned with the full transcript. The script [`evaluate_pipeline_en.py`](./src/evaluate_pipeline_en.py) and [`evaluate_pipeline_de.py](./src/evaluate_pipeline_de.py) evaluate a pipeline for English resp. German samples of audio/text by splitting the audio into voiced segments, transcribing each voiced segment with the simplified model and performing a global alignment of all inferences with the full transcript using the [Needleman-Wunsch algorithm](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm).

The resulting alignments are then evaluated by measuring...

* ... how similar each transcript and its alignment are. The [Levenshtein Similiarity](https://en.wikipedia.org/wiki/Levenshtein_distance) is used for this. The average over all test samples is calculated and stored as _Precision_
* ... how much of the full transcript is covered by the alignments. The average over all test samples is used and stored as _Recall_

_Precision_ and _Recall_ are combined to a single metric called [F-Score](https://en.wikipedia.org/wiki/F1_score). The alignments produced by the pipeline using the simplified model are also compared to the alignments produced by the pipeline using a reference model. For English the reference model is the pre-trained model that [can be downloaded from _DeepSpeech_](https://github.com/mozilla/DeepSpeech#getting-the-pre-trained-model). Since there is no reference model for German, the alignment is compared to the alignment given by the segmentation information contained in the _ReadyLingua_ metadata.

### Visualization of alignments

Evaluation will produce the alignments for each test sample in a separate folder in the target directory. This folder will contain a HTML page that can be used to play the audio file. The aligned parts will be highlighted as the audio is being played. The target directory itself contains an index file linking to each alignment and an executable script `start_server.py` that can be run to spin off a server.

## Bash Scripts

This repository contains some bash scripts that can be used to run pre-defined configurations. Note that you might have to adjust some paths in some of the scripts:

* [`create_lm.sh`](./create_lm.sh): Create a n-gram Language Model (LM) using KenLM. The LM is trained on a text corpus created from Wikipedia articles in the desired language. A vocabulary of the most common words from the corpus is also saved.
* [`learning_curve.sh`](./learning_curve.sh): Create a learning curve by training a simplified model on varying amounts of training data and evaluating each training run on a test set.
* [`evaluate_pipeline_en.sh`](./evaluate_pipeline_en.sh): Evaluate the pipeline using the simplified model trained to recognize German. The pipeline is run twice, once with the simplified model and once with the reference model. Alignment is performed on the *normalized* transcript.
* [`evaluate_pipeline_en_unnormalized.sh`](./evaluate_pipeline_en_unnormalized.sh): Evaluate the pipeline using the simplified model trained to recognize German. The pipeline is run twice, once with the simplified model and once with the reference model. Alignment is performed on the *unnormalized* transcript.
* [`evaluate_pipeline_de.sh`](./evaluate_pipeline_de.sh): Evaluate the pipeline using the simplified model trained to recognize German. The pipeline is run only once on the simplified model and the alignments are compared to the ones from _ReadyLingua_. Alignment is performed on the *normalized* transcript.
* [`evaluate_pipeline_de_unnormalized.sh`](./evaluate_pipeline_de_unnormalized.sh): Evaluate the pipeline using the simplified model trained to recognize German. The pipeline is run only once on the simplified model and the alignments are compared to the ones from _ReadyLingua_. Alignment is performed on the *unnormalized* transcript.

## Jupyter Notebooks

Some Jupyter Notebooks are available to illustrate some of the code in action. Simply run `jupyter notebook` form the root folder of this project. The webpage [http://localhost:8888/tree](http://localhost:8888/tree) should open automatically. Navigate to one of the following notebooks:

* [deep_speech_example.ipynb](./src/deep_speech_example.ipynb): Infer transcripts for audio files using the pre-trained _DeepSpeech_ model.
* [explore_corpora.ipynb](./src/explore_corpora.ipynb): Show some information on how the corpora were created, some statistics (e.g. number of audio samples per subset) and where to get the raw data from
* [explore_lm.ipynb](./src/explore_lm.ipynb): Show how a LM can be used as a simple spell checker and as a word-predictor
* [explore_needle_wundsch.ipynb](./src/explore_needle_wunsch.ipynb): Show how the _Needleman-Wunsch_ algorithm performs a global sequence alignment
* [explore_synthetization.ipynb](./src/explore_synthetization.ipynb): Show how original data can be augmented with synthesized data by applying some distortion to the original signal.
* [explore_pipeline.ipynb](./src/explore_pipeline.ipynb): Show how the pipeline works by supplying your own combination of audio/text