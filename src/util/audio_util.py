"""
Utility functions for audio manipulation
"""
import logging
import random
import wave

import librosa
import numpy as np
import soundfile as sf
from librosa.effects import time_stretch, pitch_shift
from pydub import AudioSegment

log = logging.getLogger(__name__)


def read_audio(file_path, resample_rate=None, to_mono=False, pcm16=False):
    """
    Read audio data from arbitrary files and formats with optional resampling and conversion to mono
    :param file_path: absolute path to audio file (MP3 or WAV)
    :param resample_rate: target sampling rate
    :param to_mono: if True the number of channels will be reduced to 1
    :param subformat: Subformat to use. By default a raw wave file (PCM subformat, 16-bit Integers) is assumed.
    This is the default format used in ASR (e.g. DeepSpeech). Because Librosa uses Floats by default (IEEE_FLOAT
    subformat), which is not supported by DeepSpeech.
    Therefore, if the desired subformat is PCM-16 and the source file is not a WAV file or a WAV file in a different
    subformat, the audio data will be converted on-the-fly by writing a temporary file, which slows down the loading process

    Librosa advanced use cases: https://librosa.github.io/librosa/ioformats.html
    Info about WAV-subformats: https://www.videolan.org/developers/vlc/doc/doxygen/html/vlc__codecs_8h.html

    :return: audio data in the desired subformat and sampling rate
    """
    if pcm16:
        fin = wave.open(file_path, 'rb')
        rate = fin.getframerate()
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    # no PCM-16 conversion required: Load with librosa in default format
    else:
        audio, rate = librosa.load(file_path, sr=resample_rate, mono=to_mono)
    return audio, rate


def write_audio(file_path, audio, rate, subformat='PCM_16'):
    if subformat == 'PCM_16':
        # write data to PCM-16 WAV
        sf.write(file_path, audio, rate, subtype='PCM_16')
    else:
        librosa.output.write_wav(file_path, audio, rate)


def crop_to_segments(audio, rate, segments):
    """
    Crop audio signal to match a list of segments. Leading audio frames will be cut off (cropped) until the start frame
    of the first segment. Trailing audio frames will be cut off from the end frame of the last segment.
    Segment start and end frames will be shifted to make up for the cropping

    :param audio: numpy ndarray
        audio signal as numpy array
    :param rate: int
        sampling rate
    :param segments: [corpus.corpus_segment.Segment]
        segments to use for cropping (no sorting needed)
    :return: cropped audio, sampling rate and shifted segments
    """
    crop_start = min(segment.start_frame for segment in segments)
    crop_end = max(segment.end_frame for segment in segments)

    for segment in segments:
        segment.start_frame -= crop_start
        segment.end_frame -= crop_start

    return audio[crop_start:crop_end], rate, segments


def calculate_crop(segments):
    crop_start = min(segment.start_frame for segment in segments)
    crop_end = max(segment.end_frame for segment in segments)
    return crop_start, crop_end


def read_pcm16_wave(file_path):
    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_pcm16_wave(path, audio, sample_rate):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


def resample_frame(audio_frame, old_sampling_rate=44100, new_sampling_rate=16000):
    return int(audio_frame * new_sampling_rate // old_sampling_rate)


def seconds_to_frame(time_s, sampling_rate=16000):
    return int(float(time_s) * sampling_rate)


def ms_to_frames(val_ms, sample_rate):
    return int(round(val_ms * sample_rate / 1e3))


def frame_to_ms(val_frame, sample_rate):
    return float(val_frame / sample_rate)


def shift(audio, max_shift=None):
    max_shift = max_shift if max_shift else int(0.01 * len(audio))
    shift = np.random.randint(low=1, high=max_shift)
    return audio[shift:]


def distort(audio, rate, tempo=False, pitch=False):
    audio = audio.astype(np.float32)
    distorted = audio
    if tempo:
        factor = random.uniform(0.8, 1.2) if isinstance(tempo, bool) else tempo
        distorted = time_stretch(distorted, factor)
    if pitch:
        factor = random.uniform(1, 4) if isinstance(tempo, bool) else tempo
        distorted = pitch_shift(distorted, rate, factor)
    return distorted


def mp3_to_wav(infile, outfile, outrate=16000, outchannels=1):
    AudioSegment.from_mp3(infile) \
        .set_frame_rate(outrate) \
        .set_channels(outchannels) \
        .export(outfile, format="wav")
