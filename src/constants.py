import os

from os.path import abspath, dirname, join

SRC_DIR = dirname(abspath(__file__))  # absolute path to project ./src/ directory
ROOT_DIR = abspath(join(SRC_DIR, os.pardir))  # absolute path to project root directoryy

LS_SOURCE = os.environ.get('LS_SOURCE')  # path to LibriSpeech source files
LS_TARGET = os.environ.get('LS_TARGET')  # path where LibriSpeech corpus is stored
RL_SOURCE = os.environ.get('RL_SOURCE')  # path to ReadyLingua source files
RL_TARGET = os.environ.get('RL_TARGET')  # path where ReadyLingua corpus is stored

# default parameters for training
DEFAULT_CORPUS = 'rl'
DEFAULT_LANGUAGE = 'en'
DEFAULT_N_EPOCHS = 20
DEFAULT_N_STEPS_TRAIN = 0
DEFAULT_N_STEPS_VAL = 0
DEFAULT_BATCH_SIZE = 5
DEFAULT_FEATURE_TYPE = 'mfcc'
DEFAULT_ARCHITECTURE = 'ds1'
