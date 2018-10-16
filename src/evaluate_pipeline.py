import argparse
from os import getcwd, remove, makedirs
from os.path import abspath, exists, splitext, join

import numpy as np
import pandas as pd

from core.batch_generator import VoiceSegmentsBatchGenerator
from core.decoder import BestPathDecoder, BeamSearchDecoder
from util.asr_util import infer_batches_keras, infer_batches_deepspeech, \
    extract_best_transcript
from util.audio_util import to_wav, read_pcm16_wave, frame_to_ms
from util.lm_util import load_lm_and_vocab
from util.log_util import create_args_str, print_dataframe
from util.lsa_util import align_globally
from util.pipeline_util import create_demo
from util.rnn_util import load_keras_model, load_ds_model
from util.string_util import normalize
from util.vad_util import webrtc_voice


def main(args):
    print(create_args_str(args))
    audio_path, transcript_path, keras_path, ds_path, ds_alphabet_path, ds_trie_path, lm_path, target_dir = setup(args)

    print(f'all artefacts will be saved to {target_dir}')

    audio_bytes, rate, transcript = preprocess(audio_path, transcript_path)
    segments = vad(audio_bytes, rate)
    df_transcripts = asr(segments, rate, keras_path, ds_path, ds_alphabet_path, ds_trie_path, lm_path, target_dir)
    df_transcripts = lsa(transcript, df_transcripts, target_dir)

    df_stats = calculate_stats(df_transcripts)
    create_demo(target_dir, audio_path, transcript, df_transcripts, df_stats)

    print()
    print_dataframe(df_stats)
    print()

    stats_csv = join(target_dir, 'stats.csv')
    print(f'Saving stats to {stats_csv}')
    df_transcripts.to_csv(stats_csv)


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

    args.target_dir = abspath(args.target_dir) if args.target_dir else getcwd()
    target_dir = abspath(join(args.target_dir, splitext(audio_path)[0]))
    if not exists(target_dir):
        makedirs(target_dir)

    if not args.asr_model and not args.ds_model:
        raise ValueError(f'ERROR: either --asr_model or --ds_model must be set!')

    ds_model_path, ds_alphabet_path, ds_trie_path, keras_model_path = '', '', '', ''
    if args.ds_model:
        ds_model_path = abspath(args.ds_model)
        if not exists(ds_model_path):
            raise ValueError(f'ERROR: DeepSpeech model not found at {ds_model_path}')

        if not args.alphabet_path:
            raise ValueError('ERROR: alphabet path must be specified when using DeepSpeech model')

        ds_alphabet_path = abspath(args.alphabet_path)
        if not exists(ds_alphabet_path):
            raise ValueError(f'ERROR: alphabet not found at {ds_alphabet_path}')

        ds_trie_path = abspath(args.trie_path)
        if not exists(ds_trie_path):
            raise ValueError(f'ERROR: Trie not found at {ds_trie_path}')
    else:
        keras_model_path = abspath(args.asr_model)
        if not exists(keras_model_path):
            raise ValueError(f'ERROR: Keras model not found at {keras_model_path}')

    lm_path = abspath(args.lm_path)
    if not exists(lm_path):
        raise ValueError(f'ERROR: LM not found at {lm_path}')

    return audio_path, transcript_path, keras_model_path, ds_model_path, ds_alphabet_path, ds_trie_path, lm_path, target_dir


def preprocess(audio_path, transcript_path):
    print(f'preprocessing input')

    extension = splitext(audio_path)[-1]
    if extension not in ['.wav', '.mp3']:
        raise ValueError(f'ERROR: can only handle MP3 and WAV files!')

    if extension == '.mp3':
        print(f'converting {audio_path} to PCM-16 wav')
        tmp_file = 'audio.wav'
        to_wav(audio_path, tmp_file)
        audio_bytes, rate = read_pcm16_wave(tmp_file)
        remove(tmp_file)
    else:
        audio_bytes, rate = read_pcm16_wave(audio_path)

    with open(transcript_path, 'r') as f:
        transcript = normalize(f.read())
    return audio_bytes, rate, transcript


def vad(audio, rate):
    print(f'splitting input audio into voiced segments...', end='')
    voiced_segments = list(webrtc_voice(audio, rate))
    print(f'done! Got {len(voiced_segments)} segments.')
    return voiced_segments


def asr(voiced_segments, rate, keras_path, ds_path, ds_alphabet_path, ds_trie_path, lm_path, target_dir):
    print(f'transcribing voice segments')
    transcripts_csv = join(target_dir, 'transcripts.csv')
    if exists(transcripts_csv):
        print(f'found inferences from previous run in {transcripts_csv}')
        return pd.read_csv(transcripts_csv, header=0, index_col=0)

    if ds_path:
        print(f'loading DeepSpeech model from {ds_path}, using alphabet at {ds_alphabet_path}, '
              f'LM at {lm_path} and trie at {ds_trie_path}')
        ds_model = load_ds_model(ds_path, alphabet_path=ds_alphabet_path, lm_path=lm_path, trie_path=ds_trie_path)
        transcripts = infer_batches_deepspeech(voiced_segments, rate, ds_model)
    else:
        print(f'loading Keras model from {keras_path}')
        keras_model = load_keras_model(keras_path)
        lm, lm_vocab = load_lm_and_vocab(args.lm_path)

        batch_generator = VoiceSegmentsBatchGenerator(voiced_segments, sample_rate=rate, batch_size=16)
        decoder_greedy = BestPathDecoder(keras_model)
        decoder_beam = BeamSearchDecoder(keras_model)
        df_inferences = infer_batches_keras(batch_generator, decoder_greedy, decoder_beam, lm, lm_vocab)
        transcripts = extract_best_transcript(df_inferences)

    columns = ['transcript', 'audio_start', 'audio_end']
    df_transcripts = pd.DataFrame(index=range(len(transcripts)), columns=columns)
    for i, (voice_segment, transcript) in enumerate(zip(voiced_segments, transcripts)):
        audio_start = frame_to_ms(voice_segment.start_frame, rate)
        audio_end = frame_to_ms(voice_segment.end_frame, rate)
        df_transcripts.loc[i] = [transcript, audio_start, audio_end]

    df_transcripts.index.name = 'id'
    df_transcripts.to_csv(transcripts_csv)

    return df_transcripts, transcripts_csv


def lsa(transcript, df_transcripts, target_dir):
    if 'alignment' in df_transcripts.keys():
        print(f'transcripts are already aligned')
        return df_transcripts.replace(np.nan, '', regex=True)

    print(f'aligning transcript with {len(df_transcripts)} transcribed voice segments')
    alignments = align_globally(transcript, df_transcripts['transcript'].tolist())
    df_transcripts['alignment'] = [a['text'] for a in alignments]
    df_transcripts['text_start'] = [a['start'] for a in alignments]
    df_transcripts['text_end'] = [a['end'] for a in alignments]

    transcripts_csv = join(target_dir, 'transcripts.csv')
    print(f'saving alignments to {transcripts_csv}')
    df_transcripts.to_csv(transcripts_csv)

    return df_transcripts.replace(np.nan, '', regex=True)


def calculate_stats(alignments):
    data = [[1, 1, 1]]
    df_stats = pd.DataFrame(data=data, columns=['C', 'O', 'D'])
    return df_stats


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
    parser.add_argument('--transcript', type=str, required=False,
                        help=f'(optional) path to text file containing the transcript of the recording. If not set, '
                             f'a file with the same name as \'--audio\' ending with \'.txt\' will be assumed.')
    parser.add_argument('--run_id', type=str, required=False,
                        help=f'(optional) unique ID to identify the run. A subfolder with this mane will be created in '
                             f'the same directory where the audio file is. This is useful to compare runs on the same '
                             f'audio/transcript combination with different models. If not set, the name of the audio '
                             f'file will be used.')
    parser.add_argument('--asr_model', type=str, required=False,
                        help=f'path to Keras model to use for inference')
    parser.add_argument('--ds_model', type=str, required=False,
                        help=f'path to DeepSpeech model. If set, this will be preferred over \'--asr_model\'.')
    parser.add_argument('--lm_path', type=str,
                        help=f'path to directory with KenLM binary model to use for inference')
    parser.add_argument('--trie_path', type=str, required=False,
                        help=f'(optional) path to trie, only used when --ds_model is set')
    parser.add_argument('--alphabet_path', type=str, required=False,
                        help=f'(optional) path to file containing alphabet, only used when --ds_model is set')
    parser.add_argument('--target_dir', type=str, nargs='?', help=f'path to write stats file to')
    args = parser.parse_args()

    main(args)
