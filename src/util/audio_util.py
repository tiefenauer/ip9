"""
Utility functions for audio manipulation
"""
import logging
import subprocess
import wave
from os import remove

import numpy as np
import soundfile as sf
from librosa.effects import time_stretch, pitch_shift
from pydub import AudioSegment
from pysndfx import AudioEffectsChain

log = logging.getLogger(__name__)


def to_wav(mp3_path, wav_path):
    subprocess.call(['sox', mp3_path, '-r', '16000', '-b', '16', '-c', '1', wav_path])


def resample(audio_src, audio_dst, crop_start=0, crop_end=None):
    """
    Crop audio signal to match a list of segments. Leading audio frames will be cut off (cropped) until the start frame
    of the first segment. Trailing audio frames will be cut off from the end frame of the last segment.
    Segment start and end frames will be shifted to make up for the cropping
    :param audio_src: source audio file (any format)
    :param audio_dst: target audio file (WAV, PCM 16bit LE, mono)
    :param crop_end: start frame to crop to
    :param crop_start: end frame to crop from
    :return:
    """
    to_wav(audio_src, audio_dst)
    audio, rate = sf.read(audio_dst, start=crop_start, stop=crop_end)
    sf.write(audio_dst, audio, rate, 'PCM_16')


def crop_segments(segments):
    crop_start = min(segment['start_frame'] for segment in segments)
    crop_end = max(segment['end_frame'] for segment in segments)
    for segment in segments:
        segment['start_frame'] -= crop_start
        segment['end_frame'] -= crop_start
    return crop_start, crop_end


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


def resample_frame(audio_frame, src_rate=44100, target_rate=16000):
    return int(audio_frame * target_rate // src_rate)


def seconds_to_frame(time_s, sampling_rate=16000):
    return int(float(time_s) * sampling_rate)


def ms_to_frames(val_ms, sample_rate):
    return int(round(val_ms * sample_rate / 1e3))


def frame_to_ms(val_frame, sample_rate):
    return float(val_frame / sample_rate)


def distort_audio(audio, rate, shift_s=0, pitch_factor=0, tempo_factor=1, volume=0, echo=0):
    audio_distorted = shift(audio, rate, shift_s)
    audio_distorted = change_pitch(audio_distorted, rate, pitch_factor)
    audio_distorted = change_tempo(audio_distorted, tempo_factor)
    audio_distorted = change_volume(audio_distorted, rate, db=volume)
    audio_distorted = add_echo(audio_distorted, echo)
    return audio_distorted


def shift(audio, rate, shift_s=0, shift_left=False):
    shift_frames = int(shift_s * rate)
    return audio[shift_frames:] if shift_left else np.concatenate((np.zeros((shift_frames,)), audio))


def change_pitch(audio, rate, factor=1.0):
    return pitch_shift(audio, rate, factor)


def change_tempo(audio, factor=1.0):
    return time_stretch(audio, factor)


def change_volume(audio, rate, db=0):
    if audio.dtype is not np.int16:
        tmp = 'tmp.wav'
        sf.write(tmp, audio, rate, subtype='PCM_16')
        audio_segment = AudioSegment.from_wav(tmp)
        remove(tmp)
    else:
        audio_segment = AudioSegment(audio, frame_rate=rate, channels=1, sample_width=2)
    audio_segment += db
    return np.frombuffer(audio_segment.get_array_of_samples(), dtype=np.int16)


def add_echo(audio, echo=100):
    fx = AudioEffectsChain().reverb(reverberance=echo)
    return fx(audio)


def mp3_to_wav(infile, outfile, outrate=16000, outchannels=1):
    AudioSegment.from_mp3(infile) \
        .set_frame_rate(outrate) \
        .set_channels(outchannels) \
        .export(outfile, format="wav")
