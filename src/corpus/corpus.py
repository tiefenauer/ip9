from abc import ABC
from datetime import timedelta, datetime
from itertools import product, islice
from os.path import getmtime, dirname, basename, splitext
from time import ctime

import pandas as pd

from corpus.corpus_entry import CorpusEntry
from util.string_util import contains_numeric


class Corpus(ABC):
    """
    Base class for corpora
    """

    def __init__(self, corpus_id, name, df_path=None, df=None):
        """
        Create a new corpus holding a list of corpus entries
        :param corpus_id: unique ID
        :param name: unique corpus name
        :param corpus_entries: list of CorpusEntry instances
        """
        self.corpus_id = corpus_id
        self._name = name
        self.creation_date = None
        self.root_path = None
        if df_path:
            self.df_path = df_path
            self.df = pd.read_csv(df_path)
            self.creation_date = getmtime(df_path)
            self.root_path = dirname(df_path)
        elif df:
            self.df = df

    @property
    def entries(self):
        return self.create_entries(self.df)

    def create_entries(self, df):
        for (entry_id, subset, lang, wav), df in df.groupby(['entry_id', 'subset', 'language', 'audio_file']):
            df = df.loc[:, ('start_frame', 'end_frame', 'duration', 'transcript', 'language', 'numeric')]
            yield CorpusEntry(entry_id, self, subset, lang, wav, df)

    def __iter__(self):
        for corpus_entry in self.entries:
            yield corpus_entry

    def __getitem__(self, val):
        # access by index
        if isinstance(val, int):
            return next(islice(self.entries, val, val + 1))
        if isinstance(val, slice):
            return list(islice(self.entries, val.start, val.stop))
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
        return self.df['language'].unique().tolist()

    @property
    def keys(self):
        return self.df['entry_id'].unique().tolist()

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

    def summary(self, format=None):
        print(f"""
Corpus:        {self.name}
Root:          {self.root_path}
Index:         {self.df_path}
Creation date: {ctime(self.creation_date)}
# entries:     {len(self)}    
""")

        def abs_perc_string(value, total, unit=None):
            percent = 100 * value / total if total > 0 else 0
            if unit == 's':
                value = timedelta(seconds=value)
                # delta = timedelta(seconds=value)
                # value = delta - timedelta(microseconds=delta.microseconds)
            else:
                value = f'{value:,}'
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

            audio_all_av = timedelta(seconds=df_total['duration'].mean() if df_total['duration'].any() else 0)
            audio_train_av = timedelta(seconds=df_train['duration'].mean() if df_train['duration'].any() else 0)
            audio_dev_av = timedelta(seconds=df_dev['duration'].mean() if df_dev['duration'].any() else 0)
            audio_test_av = timedelta(seconds=df_test['duration'].mean() if df_test['duration'].any() else 0)

            return [
                n_all, audio_all, str(audio_all_av),
                n_train, audio_train, str(audio_train_av),
                n_dev, audio_dev, str(audio_dev_av),
                n_test, audio_test, str(audio_test_av)
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
        df_stats = df_stats.T

        if format and format.lower() == 'html':
            return df_stats.to_html()

        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 1000,
                               'colheader_justify', 'center',
                               'display.max_colwidth', -1):
            print(df_stats)


class ReadyLinguaCorpus(Corpus):

    def __init__(self, df_path, df=None):
        super().__init__('rl', 'ReadyLingua', df_path, df)


class LibriSpeechCorpus(Corpus):

    def __init__(self, df_path, df=None):
        super().__init__('ls', 'LibriSpeech', df_path, df)


class CommonVoiceCorpus(Corpus):

    def __init__(self, train_csv, dev_csv, test_csv):
        super().__init__('cv', 'CommonVoice')
        self.creation_date = datetime.now().timestamp()
        self.root_path = dirname(train_csv)
        self.df_path = ', '.join([train_csv, dev_csv, test_csv])
        self.df = pd.concat([
            parse_cv_index(train_csv, 'train'),
            parse_cv_index(dev_csv, 'dev'),
            parse_cv_index(test_csv, 'test')
        ])


def parse_cv_index(csv_file, subset):
    df = pd.read_csv(csv_file)
    df = df.drop(['up_votes', 'down_votes', 'age', 'gender', 'accent'], axis=1)
    df['entry_id'] = df['filename'].map(lambda f: splitext(basename(f))[0])
    df['subset'] = subset
    df['language'] = 'en'
    df['start_frame'] = 0
    df['end_frame'] = df['duration'].map(lambda d: d * 16000)
    df['numeric'] = df['text'].map(lambda t: contains_numeric(t))
    df = df.rename(columns={'filename': 'audio_file', 'text': 'transcript'})
    return df
