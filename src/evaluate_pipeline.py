import argparse
from os import remove, makedirs
from os.path import abspath, exists, splitext, join, dirname, basename

import langdetect
import numpy as np
import pandas as pd
from pattern3.metrics import levenshtein_similarity

from core.batch_generator import VoiceSegmentsBatchGenerator
from core.decoder import BestPathDecoder, BeamSearchDecoder
from util.asr_util import infer_batches_keras, infer_batches_deepspeech, \
    extract_best_transcript
from util.audio_util import to_wav, read_pcm16_wave, frame_to_ms
from util.lm_util import load_lm_and_vocab, ler_norm
from util.log_util import create_args_str, print_dataframe
from util.lsa_util import align_globally
from util.pipeline_util import create_demo
from util.rnn_util import load_keras_model, load_ds_model
from util.string_util import normalize
from util.vad_util import webrtc_voice

parser = argparse.ArgumentParser(description="""
    Evaluate the performance of a pipeline by calculating the following values for each entry in a test set:
    - C: length of unaligned text (normalized by dividing by total length of ground truth)
    - O: length of overlapping alignments (normalized by dividing by total legnth of all alignments)
    - D: average Levenshtein distance over all alignments of the current entry

    The values are averaged through division by the number of entries in the test set.
    """)
parser.add_argument('--audio', type=str, required=False,
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
parser.add_argument('--language', type=str, choices=['en', 'de'], required=False,
                    help='language to train on. '
                         'English will use 26 characters from the alphabet, German 29 (umlauts)')
parser.add_argument('--lm_path', type=str,
                    help=f'path to directory with KenLM binary model to use for inference')
parser.add_argument('--trie_path', type=str, required=False,
                    help=f'(optional) path to trie, only used when --ds_model is set')
parser.add_argument('--alphabet_path', type=str, required=False,
                    help=f'(optional) path to file containing alphabet, only used when --ds_model is set')
parser.add_argument('--target_dir', type=str, nargs='?', help=f'path to write stats file to')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))
    lang, audio_path, trans_path, keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, run_id, target_dir = setup(
        args)

    print(f'all artefacts will be saved to {target_dir}')

    audio_bytes, rate, transcript, lang = preprocess(audio_path, trans_path, lang)
    segments = vad(audio_bytes, rate)
    df_alignments = asr(lang, segments, rate, keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, target_dir)
    df_alignments = gsa(transcript, df_alignments, target_dir)

    df_stats = calculate_stats(df_alignments)
    create_demo(target_dir, audio_path, transcript, df_alignments, df_stats, demo_id=run_id)

    print()
    print_dataframe(df_stats)
    print()

    stats_csv = join(target_dir, 'stats.csv')
    print(f'Saving stats to {stats_csv}')
    df_alignments.to_csv(stats_csv)


def setup(args):
    if not args.audio:
        args.audio = input('Enter path to audio file: ')
    audio_path = abspath(args.audio)
    if not exists(audio_path):
        raise ValueError(f'ERROR: no audio file found at {audio_path}')

    if args.transcript:
        transcript_path = abspath(args.transcript)
        if not exists(transcript_path):
            raise ValueError(f'ERROR: no transcript file found at {transcript_path}')
    else:
        transcript_path = abspath(splitext(audio_path)[0] + '.txt')
        while not exists(transcript_path):
            transcript_path = abspath(args.transcript)
            args.transcript = input(f'No transcript found at {transcript_path}. Enter path to transcript file: ')

    run_id = args.run_id if args.run_id else splitext(basename(audio_path))[0]

    while not args.asr_model and not args.ds_model:
        if not args.asr_model:
            args.asr_model = input(
                'Enter path to directory containing Keras model (*.h5) or Leave blank to use DeepSpeech model: ')
        if not args.asr_model and not args.ds_model:
            args.ds_model = input('Enter path to directory containing DeepSpeech model (*.pbmm): ')
            if not args.ds_model:
                print('ERROR: either --asr_model or --ds_model must be set!')

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

        run_id += '_ds'
    else:
        keras_model_path = abspath(args.asr_model)
        if not exists(keras_model_path):
            raise ValueError(f'ERROR: Keras model not found at {keras_model_path}')
        run_id += '_keras'

    if not args.lm_path:
        args.lm_path = input('Enter path to binary file of KenLM n-gram model: ')
    lm_path = abspath(args.lm_path)
    if not exists(lm_path):
        raise ValueError(f'ERROR: LM not found at {lm_path}')

    if not args.target_dir:
        args.target_dir = input(
            'Enter directory to save results to or leave blank to save in same directory like audio file: ')
    target_root = abspath(args.target_dir) if args.target_dir else dirname(audio_path)
    target_dir = abspath(join(target_root, run_id))
    if not exists(target_dir):
        makedirs(target_dir)

    if not args.language:
        args.language = input('Enter language of audio/transcript (en or de) or leave empty to detect automatically: ')
        if args.language and args.language not in ['en', 'de']:
            raise ValueError('ERROR: Language must be either en or de')

    return args.language, audio_path, transcript_path, keras_model_path, ds_model_path, ds_alphabet_path, ds_trie_path, lm_path, run_id, target_dir


def preprocess(audio_path, transcript_path, language=None):
    print("""
    ==================================================
    PIPELINE STAGE #1 (preprocessing): Converting audio to 16-bit PCM wave and normalizing transcript 
    ==================================================
    """)

    extension = splitext(audio_path)[-1]
    if extension not in ['.wav', '.mp3']:
        raise ValueError(f'ERROR: can only handle MP3 and WAV files!')

    if extension == '.mp3':
        print(f'converting {audio_path}')
        tmp_file = 'tmp.wav'
        to_wav(audio_path, tmp_file)
        audio_bytes, rate = read_pcm16_wave(tmp_file)
        remove(tmp_file)
    else:
        audio_bytes, rate = read_pcm16_wave(audio_path)

    if rate is not 16000:
        print(f'Resampling from {rate}Hz to 16.000Hz/mono')
        # audio, rate = librosa.load(audio_path, sr=16000, mono=True)
        tmp_file = 'tmp.wav'
        to_wav(audio_path, tmp_file)
        # write_pcm16_wave(tmp_file, audio, rate)
        audio_bytes, rate = read_pcm16_wave(tmp_file)
        remove(tmp_file)

    with open(transcript_path, 'r') as f:
        transcript = normalize(f.read(), language)

    if not language:
        language = langdetect.detect(transcript)
        print(f'detected language from transcript: {language}')

    print(f"""
    ==================================================
    STAGE #1 COMPLETED: Got {len(audio_bytes)} audio samples and {len(transcript)} labels
    ==================================================
    """)
    return audio_bytes, rate, transcript, language


def vad(audio, rate):
    print("""
    ==================================================
    PIPELINE STAGE #2 (VAD): splitting input audio into voiced segments 
    ==================================================
    """)
    voiced_segments = list(webrtc_voice(audio, rate))
    print(f"""
    ==================================================
    STAGE #2 COMPLETED: Got {len(voiced_segments)} segments.
    ==================================================
    """)
    return voiced_segments


def asr(language, voiced_segments, rate, keras_path, ds_path, ds_alphabet_path, ds_trie_path, lm_path, target_dir):
    print("""
    ==================================================
    PIPELINE STAGE #3 (ASR): transcribing voice segments 
    ==================================================
    """)
    alignments_csv = join(target_dir, 'alignments.csv')
    if exists(alignments_csv):
        print(f'found inferences from previous run in {alignments_csv}')
        return pd.read_csv(alignments_csv, header=0, index_col=0)

    if ds_path:
        print(f'loading DeepSpeech model from {ds_path}, using alphabet at {ds_alphabet_path}, '
              f'LM at {lm_path} and trie at {ds_trie_path}')
        ds_model = load_ds_model(ds_path, alphabet_path=ds_alphabet_path, lm_path=lm_path, trie_path=ds_trie_path)
        transcripts = infer_batches_deepspeech(voiced_segments, rate, ds_model)
    else:
        print(f'loading Keras model from {keras_path}')
        keras_model = load_keras_model(keras_path)
        lm, lm_vocab = load_lm_and_vocab(args.lm_path)

        batch_generator = VoiceSegmentsBatchGenerator(voiced_segments, sample_rate=rate, batch_size=16,
                                                      language=language)
        decoder_greedy = BestPathDecoder(keras_model, language)
        decoder_beam = BeamSearchDecoder(keras_model, language)
        df_inferences = infer_batches_keras(batch_generator, decoder_greedy, decoder_beam, language, lm, lm_vocab)
        transcripts = extract_best_transcript(df_inferences)

    columns = ['transcript', 'audio_start', 'audio_end']
    df_alignments = pd.DataFrame(index=range(len(transcripts)), columns=columns)
    for i, (voice_segment, transcript) in enumerate(zip(voiced_segments, transcripts)):
        audio_start = frame_to_ms(voice_segment.start_frame, rate)
        audio_end = frame_to_ms(voice_segment.end_frame, rate)
        df_alignments.loc[i] = [transcript, audio_start, audio_end]

    df_alignments.index.name = 'id'
    df_alignments.to_csv(alignments_csv)

    print(f"""
    ==================================================
    STAGE #3 COMPLETED: Saved transcript to {alignments_csv}
    ==================================================
    """)
    return df_alignments


def gsa(transcript, df_alignments, target_dir):
    print("""
    ==================================================
    PIPELINE STAGE #4 (GSA): aligning partial transcripts with full transcript 
    ==================================================
    """)
    if 'alignment' in df_alignments.keys():
        print(f'transcripts are already aligned')
        return df_alignments.replace(np.nan, '', regex=True)

    print(f'aligning transcript with {len(df_alignments)} transcribed voice segments')
    alignments = align_globally(transcript, df_alignments['transcript'].tolist())
    df_alignments['alignment'] = [a['text'] for a in alignments]
    df_alignments['text_start'] = [a['start'] for a in alignments]
    df_alignments['text_end'] = [a['end'] for a in alignments]

    transcripts_csv = join(target_dir, 'transcripts.csv')
    df_alignments.to_csv(transcripts_csv)

    print(f"""
    ==================================================
    STAGE #4 COMPLETED: Added alignments to {transcripts_csv}
    ==================================================
    """)
    return df_alignments.replace(np.nan, '', regex=True)


def calculate_stats(df_alignments):
    ground_truths = df_alignments['transcript'].values
    alignments = df_alignments['alignment'].values

    ler_avg = np.mean([ler_norm(gt, al) for gt, al in zip(ground_truths, alignments)])
    similarity_avg = np.mean([levenshtein_similarity(gt, al) for gt, al in zip(ground_truths, alignments)])

    data = [
        ['average LER', str(ler_avg)],
        ['average Levenshtein similarity', str(similarity_avg)],
    ]
    df_stats = pd.DataFrame(data=data, columns=['metric', 'value'])
    return df_stats


if __name__ == '__main__':
    main(args)