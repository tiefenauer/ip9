# transcribe a few examples from the LibriSpeech corpus using the pre-trained model from Mozilla downloaded at:
# https://github.com/mozilla/DeepSpeech/releases
#
# Modified version from https://github.com/mozilla/DeepSpeech/blob/master/native_client/python/client.py

import argparse
import sys
from timeit import default_timer as timer

from deepspeech import Model, printVersions
from deepspeech.model import Model

from src.util.corpus_util import get_corpus
from src.util.log_util import create_args_str


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        printVersions()
        exit(0)


parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
parser.add_argument('--model', required=True,
                    help='Path to the model (protocol buffer binary file)')
parser.add_argument('--alphabet', required=True,
                    help='Path to the configuration file specifying the alphabet used by the network')
parser.add_argument('--lm', nargs='?',
                    help='Path to the language model binary file')
parser.add_argument('--trie', nargs='?',
                    help='Path to the language model trie file created with native_client/generate_trie')
parser.add_argument('--audio', nargs='?',
                    help='Path to the audio file to run (WAV format)')
parser.add_argument('--version', action=VersionAction,
                    help='Print version and exits')
args = parser.parse_args()

# These constants control the beam search decoder
BEAM_WIDTH = 500  # Beam width used in the CTC decoder when building candidate transcriptions
LM_WEIGHT = 1.75  # The alpha hyperparameter of the CTC decoder. Language Model weight
# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00
# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training
# Number of MFCC features to use
N_FEATURES = 26
# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def main():
    # import soundfile as sf
    # for format, format_desc in sf.available_formats().items():
    #     print(f'Format: {format} {format_desc} ')
    #     for subtype, st_desc in sf.available_subtypes().items():
    #         print(f'{subtype} {st_desc}')
    #     print()

    print(create_args_str(args))
    print(f'Loading model from file {args.model}', file=sys.stderr)
    model_load_start = timer()
    ds = Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    print(f'Loaded model in {model_load_end:.3}s.', file=sys.stderr)

    # if args.lm and args.trie:
    #     print(f'Loading language model from files {args.lm} {args.trie}', file=sys.stderr)
    #     lm_load_start = timer()
    #     ds.enableDecoderWithLM(args.alphabet, args.lm, args.trie, LM_WEIGHT, VALID_WORD_COUNT_WEIGHT)
    #     lm_load_end = timer() - lm_load_start
    #     print(f'Loaded language model in {lm_load_end:.3}s.', file=sys.stderr)

    corpus = get_corpus('ls')
    corpus_entry = corpus[0]
    for i, segment in enumerate(corpus_entry[:5]):
        audio, rate = segment.audio, segment.rate
        transcription = ds.stt(audio, rate)
        print(f'transcription: \t{transcription}')
        print(f'actual: \t\t{segment.text}')


if __name__ == '__main__':
    main()
