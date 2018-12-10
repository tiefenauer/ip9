from datetime import timedelta
from itertools import islice
from os.path import join
from pathlib import Path

import pandas as pd
import soundfile as sf

from corpus.audible import Audible
from corpus.corpus_segment import Segment
from util.string_util import normalize


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
        if isinstance(item, slice):
            return list(islice(self.segments, item.start, item.stop))
        # access by index
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
        return normalize(Path(self.transcript_path).read_text(), self.language)

    def __getstate__(self):
        # prevent caches from being pickled
        state = dict(self.__dict__)
        if '_audio' in state: del state['_audio']
        if '_rate' in state: del state['_rate']
        return state

    def summary(self, html=False):
        print(f"""
Corpus Entry: {self.id}
Subset      : {self.subset}
Language    : {self.language}
Duration    : {timedelta(seconds=self.duration)} (only voiced parts)
Audio file  : {self.audio_path}
Transcript  : {self.transcript_path}
        """)

        n_all = f'{len(self.df.index)}:,'
        s_all = timedelta(seconds=self.df['duration'].sum())

        df_numeric = self.df[self.df['numeric'] == True]
        n_num = f'{len(df_numeric.index):,}' if len(df_numeric.index) else '-'
        s_num = timedelta(seconds=df_numeric['duration'].sum()) if len(df_numeric.index) else '-'

        df_non_numeric = self.df[self.df['numeric'] == False]
        n_non_num = f'{len(df_non_numeric.index):,}' if len(df_non_numeric.index) else '-'
        s_non_num = timedelta(seconds=df_non_numeric['duration'].sum()) if len(df_non_numeric.index) else '-'

        data = [
            [n_all, s_all],
            [n_num, s_num],
            [n_non_num, s_non_num]
        ]

        index = ['total samples', 'numeric samples', 'non-numeric samples']
        columns = ['#', 'duration (hh:mm:ss)']
        df_stats = pd.DataFrame(data=data, index=index, columns=columns)

        if html:
            return df_stats.to_html()

        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 1000,
                               'colheader_justify', 'center',
                               'display.max_colwidth', -1):
            print(df_stats)
