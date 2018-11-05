from datetime import timedelta
from itertools import islice
from os.path import join
from pathlib import Path

import soundfile as sf
from tabulate import tabulate

from corpus.audible import Audible
from corpus.corpus_segment import Segment


class CorpusEntry(Audible):
    """
    Class for corpus entries containing an audio signal, a transcript, segmentation and alignment information
    """

    # cache audio and rate
    _audio = None
    _rate = 16000

    def __init__(self, entry_id, corpus, subset, language, wav_name, df):
        self.id = str(entry_id)
        self.corpus = corpus
        self.subset = subset
        self.language = language
        self.wav_name = wav_name
        self.df = df

    def __len__(self):
        return len(self.df.index)

    def __iter__(self):
        for segment in self.segments:
            yield segment

    def __getitem__(self, item):
        # access by index
        if isinstance(item, slice):
            return list(islice(self.segments, item.start, item.stop))
        return next(islice(self.segments, item, item + 1))

    @property
    def segments(self):
        return self.create_segments(self.df)

    @property
    def segments_numeric(self):
        return self.create_segments(self.df[self.df['numeric'] == True])

    @property
    def segments_not_numeric(self):
        return self.create_segments(self.df[self.df['numeric'] == False])

    @property
    def duration(self):
        return self.df['duration'].sum()

    def create_segments(self, df):
        for i, (start_frame, end_frame, duration, transcript, language, numeric) in df.iterrows():
            yield Segment(self, start_frame, end_frame, duration, transcript, language, numeric)

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
    def transcript(self):
        return Path(self.transcript_path).read_text()

    def __getstate__(self):
        # prevent caches from being pickled
        state = dict(self.__dict__)
        if '_audio' in state: del state['_audio']
        if '_rate' in state: del state['_rate']
        return state

    def summary(self, format=None):
        print(f"""
Corpus Entry: {self.id}
Subset      : {self.subset}
Language    : {self.language}
Duration    : {self.duration}
Audio file  : {self.audio_path}
Transcript  : {self.transcript_path}
# segments  : {len(self)}
        """)
        total_length = sum(seg.duration for seg in self.segments)
        l_sp_num = sum(seg.duration for seg in self.segments_numeric)
        l_sp_nnum = sum(seg.duration for seg in self.segments_not_numeric)
        table = {
            '#speech segments': (len(self), timedelta(seconds=total_length)),
            '#speech segments with numbers in transcript': (
                len(list(self.segments_numeric)), timedelta(seconds=l_sp_num)),
            '#speech segments without numbers in transcript': (
                len(list(self.segments_not_numeric)), timedelta(seconds=l_sp_nnum))
        }
        headers = ['# ', 'hh:mm:ss']
        print(tabulate([(k,) + v for k, v in table.items()], headers=headers))
        print('')
