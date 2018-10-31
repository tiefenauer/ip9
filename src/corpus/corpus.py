from abc import ABC
from datetime import timedelta
from itertools import product
from os.path import getmtime, dirname
from time import ctime

import pandas as pd

from corpus.corpus_entry import CorpusEntry


class Corpus(ABC):
    """
    Base class for corpora
    """

    def __init__(self, corpus_id, name, df_path, df=None):
        """
        Create a new corpus holding a list of corpus entries
        :param corpus_id: unique ID
        :param name: unique corpus name
        :param corpus_entries: list of CorpusEntry instances
        """
        self.corpus_id = corpus_id
        self._name = name
        self.df_path = df_path
        self.df = df if df is not None else pd.read_csv(df_path)
        self.creation_date = getmtime(df_path)
        self.entries = self.create_entries(self.df)
        self.root_path = dirname(df_path)

    def create_entries(self, df):
        for (entry_id, subset, lang, wav), df_segments in df.groupby(['entry_id', 'subset', 'language', 'audio_file']):
            yield CorpusEntry(entry_id, self, subset, lang, wav, df_segments)

    def segments(self, numeric=None):
        if numeric is True:
            return [seg for entry in self.entries for seg in entry.segments_numeric]
        elif numeric is False:
            return [seg for entry in self.entries for seg in entry.segments_not_numeric]
        return [seg for entry in self.entries for seg in entry]

    def __iter__(self):
        for corpus_entry in self.entries:
            yield corpus_entry

    def __getitem__(self, val):
        # access by index
        if isinstance(val, int) or isinstance(val, slice):
            return self.entries[val]
        # access by id
        if isinstance(val, str):
            return next(iter([entry for entry in self.entries if entry.id == val]), None)
        return None

    def __len__(self):
        return len(self.df.index)

    def __call__(self, *args, **kwargs):
        languages = kwargs['languages'] if 'languages' in kwargs else self.languages
        print(f'filtering languages={languages}')
        df = self.df[self.df['language'].isin(languages)]

        numeric = kwargs['numeric'] if 'numeric' in kwargs else False
        if numeric is False:
            print(f'filtering out speech segments with numbers in transcript')
            df = df[df['numeric'] == False]

        self.__init__(self.df_path, df)
        return self

    @property
    def name(self):
        languages = ', '.join(self.languages)
        return self._name + f' (languages: {languages})'

    @property
    def languages(self):
        return sorted(set(lang for lang in (corpus_entry.language for corpus_entry in self.entries)))

    @property
    def keys(self):
        return sorted([corpus_entry.id for corpus_entry in self.entries])

    def train_set(self, numeric=False):
        df_train = self._filter_segments('train', numeric)
        return self.create_entries(df_train)

    def dev_set(self, numeric=False):
        df_dev = self._filter_segments('dev', numeric)
        return self.create_entries(df_dev)

    def test_set(self, numeric=False):
        df_test = self._filter_segments('test', numeric)
        return self.create_entries(df_test)

    def _filter_segments(self, subset, numeric=None):
        df_subset = self.df[self.df['subset'] == subset]
        if numeric is None:
            return df_subset
        return df_subset[df_subset['numeric'] == numeric]

    def summary(self):
        print(f"""
Corpus:        {self.name}
File:          {self.df_path}
Creation date: {ctime(self.creation_date)}
# entries:     {len(self.entries)}    
        """)

        def abs_perc_string(value, total, unit=None):
            percent = 100 * value / total if total > 0 else 0
            if unit == 's':
                delta = timedelta(seconds=value)
                value = delta - timedelta(microseconds=delta.microseconds)
            return f'{value} ({percent:.2f}%)'

        def create_row(df_total):
            df_train = df_total[df_total['subset'] == 'train']
            df_dev = df_total[df_total['subset'] == 'dev']
            df_test = df_total[df_total['subset'] == 'test']

            n_all = abs_perc_string(len(df_total), len(df_total))
            n_train = abs_perc_string(len(df_train), len(df_total))
            n_dev = abs_perc_string(len(df_dev), len(df_total))
            n_test = abs_perc_string(len(df_test), len(df_total))

            s_all_sum = df_total['duration'].sum()
            s_train_sum = df_train['duration'].sum()
            s_dev_sum = df_dev['duration'].sum()
            s_test_sum = df_test['duration'].sum()

            audio_all = abs_perc_string(s_all_sum, s_all_sum, unit='s')
            audio_train = abs_perc_string(s_train_sum, s_all_sum, unit='s')
            audio_dev = abs_perc_string(s_dev_sum, s_all_sum, unit='s')
            audio_test = abs_perc_string(s_test_sum, s_all_sum, unit='s')

            s_tot_mean = df_total['duration'].mean() if df_total['duration'].any() else 0
            s_train_mean = df_train['duration'].mean() if df_train['duration'].any() else 0
            s_dev_mean = df_dev['duration'].mean() if df_dev['duration'].any() else 0
            s_test_mean = df_test['duration'].mean() if df_test['duration'].any() else 0

            audio_all_av = timedelta(seconds=s_tot_mean)
            audio_train_av = timedelta(seconds=s_train_mean)
            audio_dev_av = timedelta(seconds=s_dev_mean)
            audio_test_av = timedelta(seconds=s_test_mean)

            return [
                n_all, audio_all, audio_all_av,
                n_train, audio_train, audio_train_av,
                n_dev, audio_dev, audio_dev_av,
                n_test, audio_test, audio_test_av
            ]

        data = []
        languages = self.languages + [None] if len(self.languages) > 1 else self.languages
        for lang, numeric, in product(languages, [None, True, False]):
            df_tot = self.df
            if lang:
                df_tot = df_tot[df_tot['language'] == lang]
            if numeric is not None:
                df_tot = df_tot[df_tot['numeric'] == numeric]
            data.append(create_row(df_tot))

        languages = self.languages + ['all'] if len(self.languages) > 1 else self.languages
        index = pd.MultiIndex.from_product([languages, ['all', 'numeric', 'non-numeric']])
        columns = pd.MultiIndex.from_product([['total', 'train', 'dev', 'test'], ['samples', 'audio', 'Ø audio']])
        df_stats = pd.DataFrame(data=data, index=index, columns=columns)
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 1000,
                               'colheader_justify', 'center',
                               'display.max_colwidth', -1):
            print(df_stats.T)


class ReadyLinguaCorpus(Corpus, ABC):

    def __init__(self, df_path, df=None):
        super().__init__('rl', 'ReadyLingua', df_path, df)


class LibriSpeechCorpus(Corpus):

    def __init__(self, df_path, df=None):
        super().__init__('ls', 'LibriSpeech', df_path, df)
