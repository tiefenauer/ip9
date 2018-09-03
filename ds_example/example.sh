#!/bin/bash

#positional arguments:
#  model       Path to the model (protocol buffer binary file)
#  audio       Path to the audio file to run (WAV format)
#  alphabet    Path to the configuration file specifying the alphabet used by
#              the network
#  lm          Path to the language model binary file
#  trie        Path to the language model trie file created with
#              native_client/generate_trie
#
#optional arguments:
#  -h, --help  show this help message and exit
deepspeech output_graph.pb example.wav alphabet.txt lm.binary trie