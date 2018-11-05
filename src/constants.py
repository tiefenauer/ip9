import os

from os.path import abspath, dirname, join

SRC_DIR = dirname(abspath(__file__))  # absolute path to project ./src/ directory
ROOT_DIR = abspath(join(SRC_DIR, os.pardir))  # absolute path to project root directory
ASSETS_DIR = join(ROOT_DIR, 'assets')

LS_RAW = "/media/daniel/Data/corpus/librispeech-raw"
LS_ROOT = "/media/daniel/IP9/corpora/librispeech"
RL_RAW = "/media/daniel/Data/corpus/readylingua-raw"
RL_ROOT = "/media/daniel/IP9/corpora/readylingua"
CV_ROOT = "/media/daniel/IP9/corpora/cv"

TARGET_ROOT = os.environ.get('TARGET_ROOT')

# default parameters for training
DEFAULT_CORPUS = 'rl'
DEFAULT_LANGUAGE = 'en'
DEFAULT_N_EPOCHS = 20
DEFAULT_N_STEPS_TRAIN = 0
DEFAULT_N_STEPS_VAL = 0
DEFAULT_BATCH_SIZE = 5
DEFAULT_FEATURE_TYPE = 'mfcc'
DEFAULT_ARCHITECTURE = 'ds1'
