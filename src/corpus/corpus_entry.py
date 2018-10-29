from copy import deepcopy
from datetime import timedelta
from os.path import join, splitext
from pathlib import Path

import soundfile as sf
from pydub.utils import mediainfo
from tabulate import tabulate

from corpus.audible import Audible
from corpus.corpus_segment import Segment


class CorpusEntry(Audible):
    """
    Class for corpus entries containing an audio signal, a transcript, segmentation and alignment information
    """

    # cache audio and rate
    _audio = None
    _rate = None

    def __init__(self, corpus, wav_name, df_segments):
        self.corpus = corpus
        self.language = self.corpus.language

        self.wav_name = wav_name

        self.id = splitext(wav_name)[0]
        self.audio_path = join(self.corpus.root_path, self.wav_name)
        self.transcript_path = join(self.corpus.root_path, self.id + '.txt')

        self.media_info = mediainfo(self.audio_path)
        self.segments = self.create_segments(df_segments)

    def create_segments(self, df):
        return [Segment(language=self.language,
                        start_frame=row['start_frame'],
                        end_frame=row['end_frame'],
                        transcript=row['transcript'],
                        corpus_entry=self)
                for (_, row) in df.iterrows()]

    @property
    def transcript(self):
        return Path(self.transcript_path).read_text()

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

    def __iter__(self):
        for segment in self.segments:
            yield segment

    def __getitem__(self, item):
        return self.segments[item]

    @property
    def speech_segments_numeric(self):
        return [segment for segment in self.segments if segment.contains_numeric]

    @property
    def speech_segments_not_numeric(self):
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
