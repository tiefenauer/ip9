from copy import deepcopy
from datetime import timedelta
from os.path import join
from random import randint

import numpy as np
import soundfile as sf
from tabulate import tabulate

from corpus.audible import Audible


class CorpusEntry(Audible):
    """
    Class for corpus entries containing an audio signal, a transcript, segmentation and alignment information
    """

    # cache audio and rate
    _audio = None
    _rate = None

    def __init__(self, subset, audio_):

        if parms is None:
            parms = {}
        self.corpus = None
        self.wav_name = wav_name

        for segment in segments:
            segment.corpus_entry = self
        self.speech_segments = segments

        self.raw_path = raw_path
        self.name = parms['name'] if 'name' in parms else ''
        self.id = parms['id'] if 'id' in parms else str(randint(1, 999999))
        self.language = parms['language'] if 'language' in parms else 'N/A'
        self.chapter_id = parms['chapter_id'] if 'chapter_id' in parms else 'N/A'
        self.speaker_id = parms['speaker_id'] if 'speaker_id' in parms else 'N/A'
        self.original_sampling_rate = parms['rate'] if 'rate' in parms else 'N/A'
        self.original_channels = parms['channels'] if 'channels' in parms else 'N/A'
        self.subset = parms['subset'] if 'subset' in parms else 'N/A'
        self.media_info = parms['media_info'] if 'media_info' in parms else {}

    @property
    def audio_path(self):
        return join(self.corpus.root_path, self.wav_name)

    @property
    def audio(self):
        if self._audio is not None:
            return self._audio
        self._audio, self._rate = sf.read(self.audio_path, dtype='int16')
        return self._audio

    @audio.setter
    def audio(self, audio):
        self._audio = audio.astype(np.float32)

    @property
    def rate(self):
        if self._rate is not None:
            return self._rate
        self._audio, self._rate = sf.read(self.audio_path, dtype='int16')
        return self._rate

    def __iter__(self):
        for segment in self.speech_segments:
            yield segment

    def __getitem__(self, item):
        return self.speech_segments[item]

    @property
    def speech_segments_numeric(self):
        return [segment for segment in self.speech_segments if segment.contains_numeric]

    @property
    def speech_segments_not_numeric(self):
        return [segment for segment in self.speech_segments if not segment.contains_numeric]

    @property
    def transcript(self):
        return '\n'.join(segment.transcript for segment in self.speech_segments)

    @property
    def text(self):
        return '\n'.join(segment.text for segment in self.speech_segments)

    @property
    def audio_length(self):
        return float(self.media_info['duration'])

    def __getstate__(self):
        # prevent caches from being pickled
        state = dict(self.__dict__)
        if '_audio' in state: del state['_audio']
        if '_rate' in state: del state['_rate']
        return state

    def __call__(self, *args, **kwargs):
        if not kwargs or 'include_numeric' not in kwargs or kwargs['include_numeric'] is True:
            return self
        _copy = deepcopy(self)
        segments = self.speech_segments_not_numeric
        _copy.speech_segments = segments
        _copy.name = self.name + f' ==> only segments without numeric values'
        return _copy

    def summary(self):
        print('')
        print('Corpus Entry: '.ljust(30) + f'{self.name} (id={self.id})')
        print('Audio Path: '.ljust(30) + self.wav_name)
        print('')
        total_length = sum(seg.audio_length for seg in self.speech_segments)
        l_sp_num = sum(seg.audio_length for seg in self.speech_segments_numeric)
        l_sp_nnum = sum(seg.audio_length for seg in self.speech_segments_not_numeric)
        table = {
            '#speech segments': (len(self.speech_segments), timedelta(seconds=total_length)),
            '#speech segments with numbers in transcript': (
                len(self.speech_segments_numeric), timedelta(seconds=l_sp_num)),
            '#speech segments without numbers in transcript': (
                len(self.speech_segments_not_numeric), timedelta(seconds=l_sp_nnum))
        }
        headers = ['# ', 'hh:mm:ss']
        print(tabulate([(k,) + v for k, v in table.items()], headers=headers))
        print('')
        print(f'duration: {timedelta(seconds=self.audio_length)}')
        print(f'raw path: {self.raw_path}')
        print(f'raw sampling rate: {self.original_sampling_rate}')
        print(f'raw #channels: {self.original_channels}')
        print(f'language: {self.language}')
        print(f'chapter ID: {self.chapter_id}')
        print(f'speaker_ID: {self.speaker_id}')
        print(f'subset membership: {self.subset}')
        print(f'media info: {self.media_info}')
