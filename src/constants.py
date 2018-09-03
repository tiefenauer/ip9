import os

from os.path import abspath, dirname, join

SRC_DIR = dirname(abspath(__file__))  # absolute path to project ./src/ directory
ROOT_DIR = abspath(join(SRC_DIR, os.pardir))  # absolute path to project root directoryy

# where the corpora are
CORPUS_ROOT = os.environ.get('CORPUS_ROOT') or r'E:\\' if os.name == 'nt' else '/media/all/D1'
# where to write results
TARGET_ROOT = os.environ.get('TARGET_ROOT') or r'E:\\' if os.name == 'nt' else '/media/all/D1'

# paths to corpora
RL_CORPUS_ROOT = join(CORPUS_ROOT, 'readylingua-corpus')
LS_CORPUS_ROOT = join(CORPUS_ROOT, 'librispeech-corpus')

# default parameters for training
CORPUS = 'rl'
LANGUAGE = 'en'
NUM_EPOCHS = 20
NUM_STEPS_TRAIN = 0
NUM_STEPS_VAL = 0
BATCH_SIZE = 5
FEATURE_TYPE = 'mfcc'
ARCHITECTURE = 'ds1'
