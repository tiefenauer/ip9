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