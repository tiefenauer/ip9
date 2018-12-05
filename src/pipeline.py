from functools import reduce
from genericpath import exists
from os import remove, makedirs
from os.path import splitext, join

import langdetect
import numpy as np
import pandas as pd
from keras import backend as K
from tqdm import tqdm

from core.batch_generator import VoiceSegmentsBatchGenerator
from core.decoder import BestPathDecoder, BeamSearchDecoder
from corpus.alignment import Voice
from util.asr_util import infer_batches_keras, extract_best_transcript
from util.audio_util import to_wav, read_pcm16_wave, ms_to_frames
from util.lsa_util import needle_wunsch
from util.pipeline_util import create_alignments_dataframe
from util.rnn_util import load_keras_model, load_ds_model
from util.string_util import normalize
from util.vad_util import webrtc_split


def pipeline(voiced_segments, sample_rate, transcript, language=None, keras_path=None, ds_path=None, ds_alpha_path=None,
             ds_trie_path=None, lm_path=None, lm=None, vocab=None, target_dir=None, force_realignment=False,
             align_endings=True):
    """
    Forced Alignment using pipeline.

    :param voiced_segments: list of Voice objects
    :param sample_rate: sampling rate of voiced segments
    :param language: (optional) language of the audio file. If not set, the language will be guessed from the transcript
    :param keras_path: (optional) path to directory containing Keras model (*.h5)
    :param ds_path: (optional) path to pre-trained DeepSpeech model (*.pbmm). If set, this model will be preferred
    :param keras_path: (optional) path to directory containing Keras model (*.h5)
    :param das_path: (optional) path to pre-trained DeepSpeech model (*.pbmm). If set, this model will be preferred
                    over Keras model
    :param ds_alpha_path: (optional) path to txt file containing alphabet for DS model. Required if ds_path is set
    :param ds_trie_path: (optional) path to binary file containing trie for DS model. Only used if ds_path is set
    :param lm_path: (optional) path to binary file containing KenLM n-gram Language Model
    :param vocab: (optional) path to text file containing LM vocabulary
    :param target_dir: (optional) path to directory to save results. If set, intermediate results are written and need
                       not be recalculated upon subsequent runs
    :param force_realignment: if set to True this will globally align the inferences with the transcript and override existing alignment information
    :param align_endings: align endings (not just beginnings) of partial transcripts within original transcript
    :return:
    """
    if not exists(target_dir):
        makedirs(target_dir)

    print("""PIPELINE STAGE #3 (ASR): transcribing voice segments""")
    alignments_csv = join(target_dir, 'alignments.csv')
    if target_dir and exists(alignments_csv):
        print(f'found inferences from previous run in {alignments_csv}')
        df_alignments = pd.read_csv(alignments_csv, header=0, index_col=0)
    else:
        if ds_path:
            print(f'Using DS model at {ds_path}, saving results in {target_dir}')
            transcripts = asr_ds(voiced_segments, sample_rate, ds_path, ds_alpha_path, lm_path, ds_trie_path)
        else:
            print(f'Using simplified Keras model at {keras_path}, saving results in {target_dir}')
            transcripts = asr_keras(voiced_segments, language, sample_rate, keras_path, lm, vocab)

        df_alignments = create_alignments_dataframe(voiced_segments, transcripts, sample_rate)
        if target_dir:
            print(f'saving alignments to {alignments_csv}')
            df_alignments.to_csv(join(target_dir, alignments_csv))
    df_alignments.replace(np.nan, '', regex=True, inplace=True)
    print(f"""STAGE #3 COMPLETED: Saved transcript to {alignments_csv}""")

    print("""PIPELINE STAGE #4 (GSA): aligning partial transcripts with full transcript""")
    if 'alignment' in df_alignments.keys() and not force_realignment:
        print(f'transcripts are already aligned')
    else:
        print(f'aligning transcript with {len(df_alignments)} transcribed voice segments')
        alignments = gsa(transcript, df_alignments['transcript'].tolist(), align_endings=align_endings)
        print(len(alignments))

        df_alignments['alignment'] = [a['text'] for a in alignments]
        df_alignments['text_start'] = [a['start'] for a in alignments]
        df_alignments['text_end'] = [a['end'] for a in alignments]

        if target_dir:
            print(f'saving alignments to {alignments_csv}')
            df_alignments.to_csv(alignments_csv)
    print(f"""STAGE #4 COMPLETED""")
    df_alignments.replace(np.nan, '', regex=True, inplace=True)
    return df_alignments


def preprocess(audio_path, transcript_path, language=None, norm_transcript=False):
    """
    Pipeline Stage 1: Preprocessing
    This stage prepares the input by converting it to the expected format and normalizing it

    :param audio_path: path to a MP3 or WAV file containing the speech recording
    :param transcript_path: path to a text file containing the transcript for the audio file
    :param language: (optional) hint for a language. If not set the language is detected from the transcript.
    :param norm_transcript: normalize transcript
    :return:
        raw audio bytes: the audio samples as bytes array (mono, PCM-16)
        sample rate: number of samples per second (usually 16'000)
        transcript: normalized transcript (normalization depends on language!)
        language: inferred language (if argument was omitted), else unchanged argument
    """
    print("""PIPELINE STAGE #1 (preprocessing): Converting audio to 16-bit PCM wave and normalizing transcript""")
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
        transcript = f.read()
        if norm_transcript:
            transcript = normalize(transcript)

    if not language:
        language = langdetect.detect(transcript)
        print(f'detected language from transcript: {language}')

    print(f"""STAGE #1 COMPLETED: Got {len(audio_bytes)} audio samples and {len(transcript)} labels""")
    return audio_bytes, rate, transcript, language


def vad(audio_bytes, sample_rate):
    """
    Pipeline Stage 2: Voice Activity Detection (VAD)
    This stage will take an existing audio signal and split it into voiced segments.

    :param audio_bytes: audio samples as byte array
    :param sample_rate: sampling rate
    :return: a list of voiced segments
    """
    print("""PIPELINE STAGE #2 (VAD): splitting input audio into voiced segments""")
    voiced_segments = []
    for voice_frames, voice_rate in webrtc_split(audio_bytes, sample_rate, aggressiveness=3):
        voice_bytes = b''.join([f.bytes for f in voice_frames])
        voice_audio = np.frombuffer(voice_bytes, dtype=np.int16)

        start_time = voice_frames[0].timestamp
        end_time = (voice_frames[-1].timestamp + voice_frames[-1].duration)
        start_frame = ms_to_frames(start_time * 1000, sample_rate)
        end_frame = ms_to_frames(end_time * 1000, sample_rate)
        voiced_segments.append(Voice(voice_audio, voice_rate, start_frame, end_frame))
    print(f"""STAGE #2 COMPLETED: Got {len(voiced_segments)} segments.""")
    return voiced_segments


def asr_keras(voiced_segments, lang, sample_rate, keras_path, lm, vocab):
    """
    Pipeline Stage 3: Automatic Speech Recognition (ASR) with Keras
    This stage takes a list of voiced segments and transcribes it using a simplified, self-trained Keras model

    :param voiced_segments: list of voiced segments to transcribe
    :param lang: language to use for decoding
    :param sample_rate: sampling rate of audio signals in voiced segments
    :param keras_model: absolute path to directory containing Keras model (*.h5 file)
    :param lm:
    :param vocab:
    :param gpu:
    :return: a list of transcripts for the voiced segments
    """
    keras_model = load_keras_model(keras_path)

    batch_generator = VoiceSegmentsBatchGenerator(voiced_segments, sample_rate=sample_rate, batch_size=16, lang=lang)
    decoder_greedy = BestPathDecoder(keras_model, lang)
    decoder_beam = BeamSearchDecoder(keras_model, lang)
    df_inferences = infer_batches_keras(batch_generator, decoder_greedy, decoder_beam, lang, lm, vocab)
    transcripts = extract_best_transcript(df_inferences)

    K.clear_session()
    return transcripts


def asr_ds(voiced_segments, sample_rate, ds_path, ds_alphabet_path, lm_path, ds_trie_path):
    """
    Pipeline Stage 3: Automatic Speech Recognition (ASR) with DeepSpeech
    This stage takes a list of voiced segments and transcribes it using using a pre-trained DeepSpeech (DS) model

    :param voiced_segments: list of voiced segments to transcribe
    :param ds_path: path to DS model
    :param ds_alphabet_path: path to alphabet used to train DS model
    :param lm_path: absolute path to binary file containing KenLM n-gram Language Model
    :param ds_trie_path: absolute path to file containing trie
    :param sample_rate: sample rate of the recordings
    :param ds_model: DeepSpeech model to use for inference
    :return: a list of transcripts for the voiced segments
    """
    ds = load_ds_model(ds_path, alphabet_path=ds_alphabet_path, lm_path=lm_path, trie_path=ds_trie_path)
    print('transcribing segments using DeepSpeech model')
    progress = tqdm(voiced_segments, unit=' segments')
    transcripts = []
    for voiced_segment in progress:
        transcript = ds.stt(voiced_segment.audio, sample_rate).strip()
        progress.set_description(transcript)
        transcripts.append(transcript)

    return transcripts


def gsa(transcript, partial_transcripts, align_endings):
    """
    Pipeline Stage 4: Global Sequence Alignment (GSA)
    This stage will try to align a list of partial, possibly incorrect transcripts (prediction) within a known, whole
    transcript (ground truth)

    :param transcript: whole transcript (ground truth)
    :param partial_transcripts: list of partial transcripts (predictions)
    :param align_endings: align endings (not just beginnings) of partial transcripts within original transcript
    :return:
    """
    # concatenate transcripts
    inference = ' '.join(partial_transcripts)
    # calculate boundaries [(start_index, end_index)] of partial transcripts within concatenated partial transcripts
    boundaries = reduce(
        lambda bs, t: bs + [
            ((bs[-1][1] + 1 if bs else 0), (bs[-1][1] + len(t) + 1 if bs else len(t)))],
        partial_transcripts, [])
    return needle_wunsch(transcript.lower(), inference, boundaries, align_endings=align_endings)
