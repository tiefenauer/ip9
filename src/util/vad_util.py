"""
Utility functions for VAD stage
"""
import collections
from os import remove

import librosa
import numpy as np
import soundfile as sf
from webrtcvad import Vad

from corpus.alignment import Voice


def librosa_voice(audio, rate, top_db=30, limit=None):
    intervals = librosa.effects.split(audio, top_db=top_db)
    for start, end in intervals[:limit]:
        yield Voice(audio, rate, start, end)


def webrtc_split(audio, rate, aggressiveness=3, frame_duration_ms=30, window_duration_ms=300):
    # adapted from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    audio_bytes, audio_rate = to_pcm16(audio, rate)

    vad = Vad(aggressiveness)
    num_window_frames = int(window_duration_ms / frame_duration_ms)
    sliding_window = collections.deque(maxlen=num_window_frames)
    triggered = False

    voiced_frames = []
    for frame in generate_frames(audio_bytes, audio_rate, frame_duration_ms):
        is_speech = vad.is_speech(frame.bytes, audio_rate)
        sliding_window.append((frame, is_speech))

        if not triggered:
            num_voiced = len([f for f, speech in sliding_window if speech])
            if num_voiced > 0.9 * sliding_window.maxlen:
                triggered = True
                voiced_frames += [frame for frame, _ in sliding_window]
                sliding_window.clear()
        else:
            voiced_frames.append(frame)
            num_unvoiced = len([f for f, speech in sliding_window if not speech])
            if num_unvoiced > 0.9 * sliding_window.maxlen:
                triggered = False
                yield voiced_frames, audio_rate
                sliding_window.clear()
                voiced_frames = []
    if voiced_frames:
        yield voiced_frames, audio_rate


class Frame(object):
    """
    object holding the audio signal of a fixed time interval (30ms) inside a long audio signal
    """

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def generate_frames(audio, sample_rate, frame_duration_ms=30):
    frame_length = int(sample_rate * frame_duration_ms / 1000) * 2
    offset = 0
    timestamp = 0.0
    duration = (float(frame_length) / sample_rate) / 2.0
    while offset + frame_length < len(audio):
        yield Frame(audio[offset:offset + frame_length], timestamp, duration)
        timestamp += duration
        offset += frame_length


def to_pcm16(audio, rate):
    """
    convert audio signal to PCM_16 that can be understood by WebRTC-VAD
    :param audio: audio signal (arbitrary source format, number of channels or encoding)
    :param rate: sampling rate (arbitrary value)
    :return: a PCM16-Encoded Byte array of the signal converted to 16kHz (mono)
    """
    if hasattr(audio, 'decode'):
        # Audio is already a byte string. No conversion needed
        return audio, rate

    conversion_needed = False
    if rate != 16000:
        print(f'rate {rate} does not match expected rate (16000)! Conversion needed...')
        conversion_needed = True
    if audio.ndim > 1:
        print(f'number of channels is {audio.ndim}. Audio is not mono! Conversion needed...')
        conversion_needed = True
    if audio.dtype not in ['int16', np.int16]:
        print(f'Data type is {audio.dtype}. Encoding is not PCM-16! Conversion needed...')
        conversion_needed = True

    if not conversion_needed:
        print(f'Data already conforms to expected PCM-16 format (mono, 16000 samples/s, 16-bit signed integers '
              f'(little endian). No conversion needed.')
        return audio, rate

    print(f'Data does not conform to expected PCM-16 format (mono, 16000 samples/s, 16-bit signed integers '
          f'(little endian). Conversion needed.')
    tmp_file = 'tmp.wav'
    sf.write(tmp_file, audio, rate, subtype='PCM_16')
    audio, rate = sf.read(tmp_file, dtype='int16')
    remove(tmp_file)
    return audio, rate
