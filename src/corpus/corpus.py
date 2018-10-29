from abc import ABC, abstractmethod
from datetime import timedelta
from os.path import getmtime, dirname

import numpy as np
import pandas as pd
from tabulate import tabulate

from corpus.corpus_entry import CorpusEntry


class Corpus(ABC):
    """
    Base class for corpora
    """

    def __init__(self, corpus_id, name, language, df_path):
        """
        Create a new corpus holding a list of corpus entries
        :param corpus_id: unique ID
        :param df_path: path to CSV file
        """
        self.language = language
        self.corpus_id = corpus_id
        self.name = name
        self.df_path = df_path
        self.root_path = dirname(self.df_path)
        self.entries = self.create_entries(pd.read_csv(df_path))

    def create_entries(self, df):
        return [CorpusEntry(self, audio_file, segments) for audio_file, segments in df.groupby('audio_file')]

    def segments(self, numeric=None):
        if numeric is True:
            return [seg for entry in self.entries for seg in entry.speech_segments_numeric]
        elif numeric is False:
            return [seg for entry in self.entries for seg in entry.speech_segments_not_numeric]
        return [seg for entry in self.entries for seg in entry]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, val):
        # access by index
        if isinstance(val, int) or isinstance(val, slice):
            return self.entries[val]
        # access by id
        if isinstance(val, str):
            return next(iter([entry for entry in self.entries if entry.id == val]), None)
        return None

    @property
    def keys(self):
        return sorted([corpus_entry.id for corpus_entry in self.entries])

    @abstractmethod
    def train_dev_test_split(self, include_numeric=False):
        """
        return training -, validation - and test - set
        Since these sets are constructed
        """
        pass

    def summary(self):
        creation_date = getmtime(self.df_path)
        print(f"""
        Corpus: {self.name} (id={self.corpus_id})
        File: {self.df_path}
        Creation date: {creation_date}
        # entries: {len(self.entries)}
        """)
        print('-------------------------------------')

        total = np.array([seg.audio_length for seg in self.segments()])
        numeric = np.array([seg.audio_length for seg in self.segments(numeric=True)])
        not_numeric = np.array([seg.audio_length for seg in self.segments(numeric=False)])

        total_mean = total.mean() if total.size else 0
        numeric_mean = numeric.mean() if numeric.size else 0
        not_numeric_mean = not_numeric.mean() if not_numeric.size else 0

        index = ['#segments', 'length', 'avg. length']
        columns = ['total', 'numeric', 'non-numeric']
        data = [
            [len(total), len(numeric), len(not_numeric)],
            [timedelta(seconds=total.sum()), timedelta(seconds=numeric.sum()), timedelta(seconds=not_numeric.sum())],
            [timedelta(seconds=total_mean), timedelta(seconds=numeric_mean), timedelta(seconds=not_numeric_mean)]
        ]
        df = pd.DataFrame(index=index, columns=columns, data=data)
        print(tabulate(df, headers='keys'))


class ReadyLinguaCorpus(Corpus, ABC):

    def __init__(self, language, df_path):
        super().__init__('rl', 'ReadyLingua', language, df_path)

    def train_dev_test_split(self, include_numeric=False):
        if include_numeric:
            segments = [seg for entry in self.entries for seg in entry.speech_segments]
        else:
            segments = [seg for entry in self.entries for seg in entry.speech_segments_not_numeric]

        total_length = sum(segment.audio_length for segment in segments)
        train_split = self.get_index_for_audio_length(segments, total_length * 0.8)
        test_split = self.get_index_for_audio_length(segments, total_length * 0.9)
        return segments[:train_split], segments[train_split:test_split], segments[test_split:]

    @staticmethod
    def get_index_for_audio_length(segments, min_length):
        """
        get index to split speech segments at a minimum audio length.Index will not split segments of same corpus entry
        :param segments: list of speech segments
        :param min_length: minimum audio length to split
        :return: first index where total length of speech segments is equal or greater to minimum legnth
        """
        audio_length = 0
        prev_corpus_entry_id = None
        for ix, segment in enumerate(segments):
            audio_length += segment.audio_length
            if audio_length > min_length and segment.corpus_entry.id is not prev_corpus_entry_id:
                return ix
            prev_corpus_entry_id = segment.corpus_entry.id


class LibriSpeechCorpus(Corpus):

    def __init__(self, df_path):
        super().__init__('ls', 'LibriSpeech', 'en', df_path)

    def train_dev_test_split(self, include_numeric=False):
        train_entries = filter_corpus_entry_by_subset_prefix(self.entries, 'train-')
        dev_entries = filter_corpus_entry_by_subset_prefix(self.entries, 'dev-')
        test_entries = filter_corpus_entry_by_subset_prefix(self.entries, ['test-', 'unknown'])

        if include_numeric:
            train_set = [seg for corpus_entry in train_entries for seg in corpus_entry.speech_segments]
            dev_set = [seg for corpus_entry in dev_entries for seg in corpus_entry.speech_segments]
            test_set = [seg for corpus_entry in test_entries for seg in corpus_entry.speech_segments]
        else:
            train_set = [seg for corpus_entry in train_entries for seg in corpus_entry.speech_segments_not_numeric]
            dev_set = [seg for corpus_entry in dev_entries for seg in corpus_entry.speech_segments_not_numeric]
            test_set = [seg for corpus_entry in test_entries for seg in corpus_entry.speech_segments_not_numeric]

        return train_set, dev_set, test_set


def filter_corpus_entry_by_subset_prefix(corpus_entries, prefixes):
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    return [corpus_entry for corpus_entry in corpus_entries
            if corpus_entry.subset
            and any(corpus_entry.subset.startswith(prefix) for prefix in prefixes)]
