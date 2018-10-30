from copy import deepcopy
from datetime import timedelta
from os.path import join, splitext
from pathlib import Path

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

    def __init__(self, subset, language, wav_name, segments):
        self.subset = subset
        self.language = language
        self.wav_name = wav_name
        self.corpus = None

        self.id = splitext(wav_name)[0]

        for segment in segments:
            segment.corpus_entry = self
        self.segments = segments

    @property
    def audio_path(self):
        return join(self.corpus.root_path, self.wav_name)

    @property
    def transcript_path(self):
        return join(self.corpus.root_path, self.id + '.txt')

    @property
    def audio(self):
        if self._audio is not None:
            return self._audio
        self._audio, self._rate = sf.read(self.audio_path, dtype='int16')
        return self._audio

    @property
    def rate(self):
        if self._rate is not None:
            return self._rate
        self._audio, self._rate = sf.read(self.audio_path, dtype='int16')
        return self._rate

    @property
    def audio_length(self):
        return len(self.audio) / float(self.rate)

    @property
    def transcript(self):
        return Path(self.transcript_path).read_text()

    def __iter__(self):
        for segment in self.segments:
            yield segment

    def __getitem__(self, item):
        return self.segments[item]

    @property
    def segments_numeric(self):
        return [segment for segment in self.segments if segment.contains_numeric]

    @property
    def segments_not_numeric(self):
        return [segment for segment in self.segments if not segment.contains_numeric]

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
        segments = self.segments_not_numeric
        _copy.segments = segments
        _copy.name = self.name + f' ==> only segments without numeric values'
        return _copy

    def summary(self):
        print('')
        print('Corpus Entry: '.ljust(30) + f'{self.name} (id={self.id})')
        print(f'subset: {self.subset}')
        print(f'language: {self.language}')
        print('Audio Path: '.ljust(30) + self.wav_name)
        print(f'duration: {timedelta(seconds=self.audio_length)}')
        print('')
        total_length = sum(seg.audio_length for seg in self.segments)
        l_sp_num = sum(seg.audio_length for seg in self.segments_numeric)
        l_sp_nnum = sum(seg.audio_length for seg in self.segments_not_numeric)
        table = {
            '#speech segments': (len(self.segments), timedelta(seconds=total_length)),
            '#speech segments with numbers in transcript': (
                len(self.segments_numeric), timedelta(seconds=l_sp_num)),
            '#speech segments without numbers in transcript': (
                len(self.segments_not_numeric), timedelta(seconds=l_sp_nnum))
        }
        headers = ['# ', 'hh:mm:ss']
        print(tabulate([(k,) + v for k, v in table.items()], headers=headers))
        print('')
