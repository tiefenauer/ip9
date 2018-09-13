# IP9
Code for my master thesis at FHNW

## Setup
The code relies on the following environmental variables being set:

| Name | Description |
|---|---|
| LS_SOURCE | Absolute path to directory where LibriSpeech raw data is stored |
| RL_SOURCE | Absolute path to directory where ReadyLingua raw data is stored |
| LS_TARGET | Absolute path to directory where LibriSpeech corpus data is stored |
| RL_TARGET | Absolute path to directory where ReadyLingua corpus data is stored |

The `xx_TARGET` directories must provide enough space to store the processed files. Their content is changed when recreating the corpora.

### Dependencies

* This application uses [Pydub](http://pydub.com/) which means you will need **libav** or **ffmpeg** on your `PATH`. See the [Pydub Repository](https://github.com/jiaaro/pydub#installation) for further instructions.
* Visual C++ build tools to work with webrtcvad (google it for download link): must be installed before installing the python requirements (see below)!
* [Sox](http://sox.sourceforge.net/) is used to convert MP3 into PCM-WAV and must be on the `PATH`

### Installation
1. Clone [the repository](https://github.com/tiefenauer/ip9): `git clone git@github.com:tiefenauer/ip9.git` 
2. Install Python requirements: `pip install -r requirements.txt`
3. Install [TensorFlow](https://www.tensorflow.org/install/): TF is not included in `requirements.txt` because you can choose between the `tensorflow` (no GPU acceleration) and `tensorflow-gpu` (with GPU-acceleration). If your computer does not have a CUDA-supported GPU (like mine does) you will install the former, else the latter. Installing `tensorflow-gpu` on a computer without GPU does not work (at least I did not get it to work).

## Creating the corpora

To train the Neural Network for ASR, corpora must be created. The raw data is read from the source directory containing raw data and is written to the target directory. All audio data will be converted to RAW-WAV (PCM) using 16-bit signed Integer encoding (little endian). The following table shows the most important information to create each corpus. Type `python {filename} -h` to see options.

| corpus | file to create corpus | source dir (raw data) | target dir (processed data) | estimated time needed |
|---|---|---|---|---|
| LibriSpeech | `create_ls_corpus.py` | `$LS_SOURCE` | `$LS_TARGET` | 6-7h |
| ReadyLingua | `create_rl_corpus.py` | `$RL_SOURCE` | `$RL_TARGET` | 2-3m |
