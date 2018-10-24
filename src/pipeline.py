from functools import reduce
from os import remove
from os.path import splitext

import langdetect
import numpy as np
from tqdm import tqdm

from core.batch_generator import VoiceSegmentsBatchGenerator
from core.decoder import BestPathDecoder, BeamSearchDecoder
from corpus.alignment import Voice
from util.asr_util import infer_batches_keras, extract_best_transcript
from util.audio_util import to_wav, read_pcm16_wave, ms_to_frames
from util.lm_util import load_lm_and_vocab
from util.lsa_util import needle_wunsch
from util.rnn_util import load_ds_model, load_keras_model
from util.string_util import normalize
from util.vad_util import webrtc_split


def preprocess(audio_path, transcript_path, language=None):
    """
    Pipeline Stage 1: Preprocessing
    This stage prepares the input by converting it to the expected format and normalizing it

    :param audio_path: path to a MP3 or WAV file containing the speech recording
    :param transcript_path: path to a text file containing the transcript for the audio file
    :param language: (optional) hint for a language. If not set the language is detected from the transcript.
    :return:
        raw audio bytes: the audio samples as bytes array (mono, PCM-16)
        sample rate: number of samples per second (usually 16'000)
        transcript: normalized transcript (normalization depends on language!)
        language: inferred language (if argument was omitted), else unchanged argument
    """
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

    return audio_bytes, rate, transcript, language


def vad(audio_bytes, sample_rate):
    """
    Pipeline Stage 2: Voice Activity Detection (VAD)
    This stage will take an existing audio signal and split it into voiced segments.

    :param audio_bytes: audio samples as byte array
    :param sample_rate: sampling rate
    :return: a list of voiced segments
    """
    voiced_segments = []
    for voice_frames, voice_rate in webrtc_split(audio_bytes, sample_rate, aggressiveness=3):
        voice_bytes = b''.join([f.bytes for f in voice_frames])
        voice_audio = np.frombuffer(voice_bytes, dtype=np.int16)

        start_time = voice_frames[0].timestamp
        end_time = (voice_frames[-1].timestamp + voice_frames[-1].duration)
        start_frame = ms_to_frames(start_time * 1000, sample_rate)
        end_frame = ms_to_frames(end_time * 1000, sample_rate)
        voiced_segments.append(Voice(voice_audio, voice_rate, start_frame, end_frame))
    return voiced_segments


def asr_keras(voiced_segments, language, sample_rate, keras_path, lm_path):
    """
    Pipeline Stage 3: Automatic Speech Recognition (ASR) with Keras
    This stage takes a list of voiced segments and transcribes it using a simplified, self-trained Keras model

    :param voiced_segments: list of voiced segments to transcribe
    :param language: language to use for decoding
    :param sample_rate: sampling rate of audio signals in voiced segments
    :param keras_path: absolute path to directory containing Keras model (*.h5 file)
    :param lm_path: absolute path to binary file containing KenLM n-gram Language Model
    :return: a list of transcripts for the voiced segments
    """
    keras_model = load_keras_model(keras_path)
    lm, lm_vocab = load_lm_and_vocab(lm_path)

    batch_generator = VoiceSegmentsBatchGenerator(voiced_segments, sample_rate=sample_rate, batch_size=16, language=language)
    decoder_greedy = BestPathDecoder(keras_model, language)
    decoder_beam = BeamSearchDecoder(keras_model, language)
    df_inferences = infer_batches_keras(batch_generator, decoder_greedy, decoder_beam, language, lm, lm_vocab)
    transcripts = extract_best_transcript(df_inferences)

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


def gsa(transcript, partial_transcripts):
    """
    Pipeline Stage 4: Global Sequence Alignment (GSA)
    This stage will try to align a list of partial, possibly incorrect transcripts (prediction) within a known, whole
    transcript (ground truth)

    :param transcript: whole transcript (ground truth)
    :param partial_transcripts: list of partial transcripts (predictions)
    :return:
    """
    inference = ' '.join(partial_transcripts)
    beginnings = reduce(lambda x, y: x + [len(y) + x[-1] + 1], partial_transcripts[:-1], [0])
    return needle_wunsch(transcript, inference, beginnings)
