import argparse
import math
from itertools import product
from os import getcwd, remove, makedirs
from os.path import abspath, exists, splitext, join

import numpy as np
import pandas as pd

from core.batch_generator import VoiceSegmentsBatchGenerator
from core.decoder import BestPathDecoder, BeamSearchDecoder
from util.asr_util import infer_batches, decoding_strategies, lm_uses
from util.audio_util import to_wav, read_pcm16_wave
from util.lm_util import load_lm
from util.log_util import create_args_str
from util.lsa_util import align_globally
from util.rnn_util import load_model_from_dir
from util.string_util import normalize
from util.vad_util import webrtc_voice


def main(args):
    print(create_args_str(args))
    audio_path, transcript_path, asr_model, lm, lm_vocab, target_dir = setup(args)

    print(f'all artefacts will be saved to {target_dir}')

    audio, rate, transcript = preprocess(audio_path, transcript_path)
    voiced_segments = vad(audio, rate)
    transcripts = asr(asr_model, voiced_segments, rate, lm, lm_vocab, target_dir)
    alignments = lsa(transcripts, transcript)

    c, o, d = calculate_stats(alignments)
    df = pd.DataFrame(index=np.arange(len(test_set)), columns=['C', 'O', 'D'])
    df.loc[ix] = [c, o, d]

    print('Stats:')
    print(f'average C:', df['C'].values.mean())
    print(f'average O:', df['O'].values.mean())
    print(f'average D:', df['D'].values.mean())
    print()
    print(f'Saving stats to {target_dir}')
    df.to_csv(target_dir)


def setup(args):
    audio_path = abspath(args.audio)
    if not exists(audio_path):
        raise ValueError(f'ERROR: no audio file found at {audio_path}')
    if args.transcript:
        transcript_path = abspath(args.transcript)
        if not exists(transcript_path):
            raise ValueError(f'ERROR: no transcript file found at {transcript_path}')
    else:
        transcript_path = abspath(splitext(audio_path)[0] + '.txt')
        if not exists(transcript_path):
            raise ValueError(f'ERROR: not transcript file supplied and no transcript found at {transcript_path}')

    asr_model_path = abspath(args.asr_model)
    if not exists(asr_model_path):
        raise ValueError(f'ERROR: ASR model not found at {asr_model_path}')
    model = load_model_from_dir(asr_model_path)

    lm_path = abspath(args.lm)
    if not exists(lm_path):
        raise ValueError(f'ERROR: LM not found at {lm_path}')
    lm, lm_vocab = load_lm(lm_path)

    args.target_dir = abspath(args.target_dir) if args.target_dir else getcwd()
    target_dir = abspath(join(args.target_dir, splitext(audio_path)[0]))
    if not exists(target_dir):
        makedirs(target_dir)

    return audio_path, transcript_path, model, lm, lm_vocab, target_dir


def preprocess(audio_path, transcript_path):
    print(f'preprocessing input')

    extension = splitext(audio_path)[-1]
    if extension not in ['.wav', '.mp3']:
        raise ValueError(f'ERROR: can only handle MP3 and WAV files!')

    if extension == '.mp3':
        print(f'converting {audio_path} to PCM-16 wav')
        tmp_file = 'audio.wav'
        to_wav(audio_path, tmp_file)
        audio, rate = read_pcm16_wave(tmp_file)
        remove(tmp_file)
    else:
        audio, rate = read_pcm16_wave(audio_path)

    with open(transcript_path, 'r') as f:
        transcript = normalize(f.read())
    return audio, rate, transcript


def vad(audio, rate):
    print(f'splitting input audio into voiced segments...', end='')
    voiced_segments = list(webrtc_voice(audio, rate))
    print(f'done! Got {len(voiced_segments)} segments.')
    return voiced_segments


def asr(model, voiced_segments, rate, lm, lm_vocab, target_dir):
    inferences_csv = join(target_dir, 'inferences.csv')
    if exists(inferences_csv):
        print(f'found inferences from previous run in {target_dir}')
        df_inferences = pd.read_csv(inferences_csv, header=[0, 1, 2], index_col=0)

    else:
        print(f'inferring partial transcripts')
        batch_generator = VoiceSegmentsBatchGenerator(voiced_segments, sample_rate=rate, batch_size=16)
        decoder_greedy = BestPathDecoder(model)
        decoder_beam = BeamSearchDecoder(model)
        df_inferences = infer_batches(batch_generator, decoder_greedy, decoder_beam, lm, lm_vocab)
        df_inferences.to_csv(inferences_csv)

    transcripts = [''] * len(df_inferences)

    for ix, row in df_inferences.iterrows():
        ler_min = math.inf
        transcript = ''
        for decoding_strategy, lm_use in product(decoding_strategies, lm_uses):
            ler_value = row[(decoding_strategy, lm_use)]['LER']
            if ler_value < ler_min:
                ler_min = ler_value
                transcript = row[(decoding_strategy, lm_use)]['prediction']
        # print(transcript)
        transcripts[ix] = transcript

    return transcripts


def lsa(transcripts, full_transcript):
    alignments = align_globally(transcripts, full_transcript)
    for alignment in alignments:
        print('original :', full_transcript[alignment['start']:alignment['end']])
        print('alignment:', alignment['text'])


def calculate_stats(alignments):
    return 1, 1, 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Evaluate the performance of a pipeline by calculating the following values for each entry in a test set:
        - C: length of unaligned text (normalized by dividing by total length of ground truth)
        - O: length of overlapping alignments (normalized by dividing by total legnth of all alignments)
        - D: average Levenshtein distance over all alignments of the current entry

        The values are averaged through division by the number of entries in the test set.
        """)
    parser.add_argument('--audio', type=str,
                        help=f'path to audio file containing a recording to be aligned (mp3 or wav)')
    parser.add_argument('--transcript', type=str,
                        help=f'(optional) path to text file containing the transcript of the recording. If not set, '
                             f'a file with the same name as \'--audio\' ending with \'.txt\' will be assumed.')
    parser.add_argument('--asr_model', type=str, help=f'path to Keras model to use for inference')
    parser.add_argument('--lm', type=str, help=f'path to directory with KenLM binary model to use for inference')
    parser.add_argument('--target_dir', type=str, nargs='?', help=f'path to write stats file to')
    args = parser.parse_args()

    main(args)
