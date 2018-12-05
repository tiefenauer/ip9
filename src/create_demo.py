import argparse
from os import makedirs
from os.path import abspath, exists, splitext, join, dirname, basename

from pipeline import pipeline, preprocess, vad
from util.lm_util import load_lm, load_vocab
from util.log_util import create_args_str, print_dataframe
from util.pipeline_util import create_demo_files, query_asr_params, calculate_stats, query_lm_params
from util.rnn_util import query_gpu

parser = argparse.ArgumentParser(description="""Create HTML page with demonstration of pipeline performance""")
parser.add_argument('--audio', type=str, required=False,
                    help=f'path to audio file containing a recording to be aligned (mp3 or wav)')
parser.add_argument('--transcript', type=str, required=False,
                    help=f'(optional) path to text file containing the transcript of the recording. If not set, '
                    f'a file with the same name as \'--audio\' ending with \'.txt\' will be assumed.')
parser.add_argument('--keras_path', type=str, required=False,
                    help=f'path to Keras model to use for inference')
parser.add_argument('--ds_path', type=str, required=False,
                    help=f'path to DeepSpeech model. If set, this will be preferred over \'--asr_model\'.')
parser.add_argument('--ds_trie_path', type=str, required=False,
                    help=f'(optional) path to trie, only used when --ds_model is set')
parser.add_argument('--ds_alpha_path', type=str, required=False,
                    help=f'(optional) path to file containing alphabet, only used when --ds_model is set')
parser.add_argument('--language', type=str, choices=['en', 'de', 'fr', 'it', 'es'], required=False,
                    help='language to train on. '
                         'English will use 26 characters from the alphabet, German 29 (+umlauts)')
parser.add_argument('--lm_path', type=str,
                    help=f'path to directory with KenLM binary model to use for inference')
parser.add_argument('--vocab_path', type=str, required=False,
                    help=f'(optional) Path to vocabulary for LM')
parser.add_argument('--target_dir', type=str, nargs='?', help=f'path to write stats file to')
parser.add_argument('--gpu', type=str, required=False, default=None,
                    help='(optional) GPU(s) to use for training. If not set, you will be asked at runtime.')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))
    lang, audio_file, transcript_file, keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, vocab_path, target_dir, gpu = setup(
        args)
    print(f'all artefacts will be saved to {target_dir}')

    lm = load_lm(lm_path) if lm_path else None
    vocab = load_vocab(vocab_path) if vocab_path else None

    audio_bytes, sample_rate, transcript, language = preprocess(audio_file, transcript_file, 'en')
    voiced_segments = vad(audio_bytes, sample_rate)
    df_alignments = pipeline(voiced_segments=voiced_segments, sample_rate=sample_rate, transcript=transcript,
                             language='en',
                             keras_path=keras_path, lm=lm, vocab=vocab,
                             force_realignment=True,
                             target_dir=target_dir)

    df_stats = calculate_stats(df_alignments, ds_path, transcript)
    create_demo_files(target_dir, audio_file, transcript, df_alignments, df_stats)

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

    keras_path, ds_path, ds_alpha_path, ds_trie_path = query_asr_params(args)
    lm_path, vocab_path = query_lm_params(args)
    gpu = args.gpu if args.gpu else query_gpu()

    if not args.target_dir:
        args.target_dir = input(
            'Enter directory to save results to or leave blank to save in same directory like audio file: ')
    target_root = abspath(args.target_dir) if args.target_dir else dirname(audio_path)

    demo_id = splitext(basename(audio_path))[0] + ('_keras' if keras_path else '_ds')
    target_dir = abspath(join(target_root, demo_id))

    if not exists(target_dir):
        makedirs(target_dir)

    if not args.language:
        args.language = input('Enter language of audio/transcript (en or de) or leave empty to detect automatically: ')
        if args.language and args.language not in ['en', 'de', 'fr', 'it', 'es']:
            raise ValueError('ERROR: Language must be either en or de')

    return args.language, audio_path, transcript_path, keras_path, ds_path, ds_alpha_path, ds_trie_path, lm_path, vocab_path, target_dir, gpu


if __name__ == '__main__':
    main(args)
