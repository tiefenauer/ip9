# IP9
Code for my master thesis at FHNW

## Setup
The code relies on corpora in the format expected by DeepSpeech. Such a corpus is usually just a directory with one PCM-16 encoded `*.wav` file for each training sample. The transcript of each training sample is contained in a `*.txt` file with the same base name. The samples are split into training-, validation- and test-set. Samples in each set are defined by a csv file with the following colums:

- `wav_filename`: Path to the wav file. Note that the path must be absolute, which means the corpus is not easily portable.
* `wav_filesize`: size of the wav file in bytes
* `wav_length`: length of the recording in seconds
* `transcript`: arbitrary string containing the transcript. The string must be lowercase and only contain characters from the alphabet the model is trained on. 

**Example CSV file for corpus**:
```csv
wav_filename,wav_filesize,wav_length,transcript
/path/to/001.wav,179882,5.6199375,some transcript
/path/to/002.wav,123402,3.8549375,some other transcript
...
```

### Dependencies

* This application uses [Pydub](http://pydub.com/) which means you will need **libav** or **ffmpeg** on your `PATH`. See the [Pydub Repository](https://github.com/jiaaro/pydub#installation) for further instructions.
* Visual C++ build tools to work with webrtcvad (google it for download link): must be installed before installing the python requirements (see below)!
* [Sox](http://sox.sourceforge.net/) is used to convert MP3 into PCM-WAV and must be on the `PATH`. You also need to install the handler for MP3 files. On Linux it this is easiest done by executing the following commants:
  * `sudo apt-get install sox`: to install Sox
  * `sudo apt-get install libsox-fmt-all`: to install all file handlers 
* [PipeViewer](http://www.ivarch.com/programs/pv.shtml) for some bash scripts
* [pygit2][https://www.pygit2.org] is used to include the commit ID when plotting results. It uses [libgit2](https://libgit2.org/), which must be installed before installing the pygit2 Python package. See [the installation instructions](https://www.pygit2.org/install.html#quick-install) on how to quickly install libgit2.
* [KenLM](https://github.com/kpu/kenlm) is used to create an n-gram Language Model. It is assumed that `lmplz` and `build_binary` are on `$PATH`. See [the KenLM docs](https://kheafield.com/code/kenlm/) for more information about how to build those binaries from source.

### Installation
1. Clone [the repository](https://github.com/tiefenauer/ip9): `git clone git@github.com:tiefenauer/ip9.git` 
2. Install Python requirements: `pip install -r requirements.txt`. **IMPORTANT: Make sure that Keras 2.2.2 is installed! (`pip show keras`)**
3. Install [TensorFlow](https://www.tensorflow.org/install/): TF is not included in `requirements.txt` because you can choose between the `tensorflow` (no GPU acceleration) and `tensorflow-gpu` (with GPU-acceleration). If your computer does not have a CUDA-supported GPU (like mine does) you will install the former, else the latter. Installing `tensorflow-gpu` on a computer without GPU does not work (at least I did not get it to work).

## Creating the corpora

To train the Neural Network for ASR, corpora must be created. The raw data is read from the source directory containing raw data and is written to the target directory. All audio data will be converted to RAW-WAV (PCM) using 16-bit signed Integer encoding (little endian). The following table shows the most important information to create each corpus. Type `python {filename} -h` to see options.

| corpus | file to create corpus | source dir (raw data) | target dir (processed data) | estimated time needed |
|---|---|---|---|---|
| LibriSpeech | `create_ls_corpus.py` | `$LS_SOURCE` | `$LS_TARGET` | 6-7h |
| ReadyLingua | `create_rl_corpus.py` | `$RL_SOURCE` | `$RL_TARGET` | 2-3m |
