import argparse
from os import getcwd
from os.path import abspath, exists

import numpy as np
import pandas as pd
import soundfile as sf
from keras.engine.saving import load_model
from pattern3.metrics import levenshtein_similarity

from util.asr_util import infer_transcription
from util.lm_util import correction, load_LM
from util.log_util import create_args_str
from util.lsa_util import smith_waterman
from util.string_util import normalize
from util.vad_util import webrtc_vad

parser = argparse.ArgumentParser(description="""
    Evaluate the performance of a pipeline by calculating the following values for each entry in a test set:
    - C: length of unaligned text (normalized by dividing by total length of ground truth)
    - O: length of overlapping alignments (normalized by dividing by total legnth of all alignments)
    - D: average Levenshtein distance over all alignments of the current entry
    
    The values are averaged through division by the number of entries in the test set.
    """)
parser.add_argument('--test_files', type=str, help=f'path to CSV file containing the entries of the test set')
parser.add_argument('--asr_model', type=str, help=f'path to Keras model to use for inference')
parser.add_argument('--lm', type=str, help=f'path to KenLM binary model to use for inference')
parser.add_argument('--output_file', type=str, nargs='?', help=f'path to write stats file to')
args = parser.parse_args()


def main():
    print(create_args_str(args))
    test_set, asr_model, lm, lm_vocab, stats_file_path = setup(args)

    df = pd.DataFrame(index=np.arange(len(test_set)), columns=['C', 'O', 'D'])

    for i, test_entry in enumerate(test_set):
        # this is the pipeline
        audio, rate, transcript = preprocess(test_entry)
        voiced_segments = vad(audio, rate)
        partial_transcripts = asr(voiced_segments, rate, asr_model, lm, lm_vocab)
        alignments = lsa(partial_transcripts, transcript)

        c, o, d = calculate_stats(alignments)
        df.loc[i] = [c, o, d]

    print('Stats:')
    print(f'average C:', df['C'].values.mean())
    print(f'average O:', df['O'].values.mean())
    print(f'average D:', df['D'].values.mean())
    print()
    print(f'Saving stats to {stats_file_path}')
    df.to_csv(stats_file_path)


def setup(args):
    test_set_path = abspath(args.test_set)
    if not exists(test_set_path):
        raise ValueError(f'ERROR: test set file not found at {test_set_path}')
    test_set = pd.read_csv(test_set_path)

    asr_model_path = abspath(args.asr_model)
    if not exists(asr_model_path):
        raise ValueError(f'ERROR: ASR model not found at {asr_model_path}')
    asr_model = load_model(asr_model_path)

    lm_path = abspath(args.lm)
    if not exists(lm_path):
        raise ValueError(f'ERROR: LM not found at {lm_path}')
    lm, lm_vocab = load_LM(lm_path)

    stats_file_path = args.stats_file if args.stats_file else getcwd()

    return test_set, asr_model, lm, lm_vocab, stats_file_path


def preprocess(test_entry):
    wav_path, transcript = test_entry['wav_filename'], test_entry['transcript']
    audio, rate = sf.read(wav_path, samplerate=16000, subtype='PCM_16')
    transcript_normalized = normalize(transcript)
    return audio, rate, transcript_normalized


def vad(audio, rate):
    speech_segments = list(webrtc_vad(audio, rate))
    return speech_segments


def asr(voiced_segments, rate, model, lm, lm_vocab):
    partial_transcripts = []
    for segment in voiced_segments:
        transcription_inferred = infer_transcription(model, segment.audio, rate)
        transcription_corrected = correction(transcription_inferred, lm, lm_vocab)
        partial_transcripts.append(transcription_corrected)
    return partial_transcripts


def lsa(partial_transcripts, full_transcript):
    alignments = []
    for partial_transcript in partial_transcripts:
        text_start, text_end, b_ = smith_waterman(partial_transcript, full_transcript)
        alignment_text = full_transcript[text_start:text_end]
        similarity = levenshtein_similarity(normalize(partial_transcript), normalize(alignment_text))
        alignment = {'text': alignment_text, 'similarity': similarity, 'text_start': text_start, 'text_end': text_end}
        alignments.append(alignment)
    return alignments


def calculate_stats(alignments):
    return 1, 1, 1


if __name__ == '__main__':
    main()
