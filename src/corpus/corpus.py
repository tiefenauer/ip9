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
            self.creation_date = getmtime(df_path)
            self.root_path = dirname(df_path)
            self.df = df if df is not None else pd.read_csv(df_path)
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
        languages = kwargs['languages'].split(',') if 'languages' in kwargs else self.languages
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
        return self._get_segments_for_subset('train', numeric)

    def dev_set(self, numeric=False):
        return self._get_segments_for_subset('dev', numeric)

    def test_set(self, numeric=False):
        return self._get_segments_for_subset('test', numeric)

    def _get_segments_for_subset(self, subset, numeric):
        df_subset = self._filter_segments(subset, numeric)
        if numeric is True:
            return [segment for entry in self.create_entries(df_subset) for segment in entry.segments_numeric]
        elif numeric is False:
            return [segment for entry in self.create_entries(df_subset) for segment in entry.segments_not_numeric]
        return [segment for entry in self.create_entries(df_subset) for segment in entry.segments]

    def _filter_segments(self, subset, numeric=None):
        df_subset = self.df[self.df['subset'] == subset]
        if numeric is None:
            return df_subset
        return df_subset[df_subset['numeric'] == numeric]

    def summary(self, html=False):
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
            else:
                value = f'{value:,}'
            return f'{value} ({percent:.2f}%)'

        def create_row(df, n_total, s_total):
            df_train = df[df['subset'] == 'train']
            df_dev = df[df['subset'] == 'dev']
            df_test = df[df['subset'] == 'test']

            n_all = abs_perc_string(len(df), n_total) if len(df) else '-'
            n_train = abs_perc_string(len(df_train), n_total) if len(df_train) else '-'
            n_dev = abs_perc_string(len(df_dev), n_total) if len(df_dev) else '-'
            n_test = abs_perc_string(len(df_test), n_total) if len(df_test) else '-'

            s_all = df['duration'].sum()
            s_train = df_train['duration'].sum()
            s_dev = df_dev['duration'].sum()
            s_test = df_test['duration'].sum()

            audio_all = abs_perc_string(s_all, s_total, unit='s') if s_all else '-'
            audio_train = abs_perc_string(s_train, s_total, unit='s') if s_train else '-'
            audio_dev = abs_perc_string(s_dev, s_total, unit='s') if s_dev else '-'
            audio_test = abs_perc_string(s_test, s_total, unit='s') if s_test else '-'

            audio_all_av = timedelta(seconds=df['duration'].mean()) if df['duration'].any() else '-'
            audio_train_av = timedelta(seconds=df_train['duration'].mean()) if df_train['duration'].any() else '-'
            audio_dev_av = timedelta(seconds=df_dev['duration'].mean()) if df_dev['duration'].any() else '-'
            audio_test_av = timedelta(seconds=df_test['duration'].mean()) if df_test['duration'].any() else '-'

            trans_all_av = f"{df['transcript'].map(len).mean():.2f}" if df['transcript'].any() else '-'
            trans_train_av = f"{df_train['transcript'].map(len).mean():.2f}" if df['transcript'].any() else '-'
            trans_dev_av = f"{df_dev['transcript'].map(len).mean():.2f}" if df['transcript'].any() else '-'
            trans_test_av = f"{df_test['transcript'].map(len).mean():.2f}" if df['transcript'].any() else '-'

            return [
                n_all, n_train, n_dev, n_test,
                audio_all, audio_train, audio_dev, audio_test,
                str(audio_all_av), str(audio_train_av), str(audio_dev_av), str(audio_test_av),
                trans_all_av, trans_train_av, trans_dev_av, trans_test_av
            ]

        data = []
        languages = self.languages + [None] if len(self.languages) > 1 else self.languages
        for lang, numeric, in product(languages, [None, True, False]):
            df = self.df
            if lang:
                df = df[df['language'] == lang]

            n_total = len(df)
            s_total = df['duration'].sum()

            if numeric is not None:
                df = df[df['numeric'] == numeric]
            data.append(create_row(df, n_total=n_total, s_total=s_total))

        languages = self.languages + ['all'] if len(self.languages) > 1 else self.languages
        index = pd.MultiIndex.from_product([languages, ['all', 'numeric', 'non-numeric']])
        columns = pd.MultiIndex.from_product([['samples', 'audio', 'Ø audio', 'Ø transcript'],
                                              ['total', 'train', 'dev', 'test']])
        df_stats = pd.DataFrame(data=data, index=index, columns=columns)
        df_stats = df_stats.T

        if html:
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


class DeepSpeechCorpus(Corpus):

    def __init__(self, language, train_csv, dev_csv, test_csv):
        super().__init__('cv', 'CommonVoice')
        self.creation_date = datetime.now().timestamp()
        self.root_path = dirname(train_csv)
        self.df_path = ', '.join([train_csv, dev_csv, test_csv])
        self.df = pd.concat([
            parse_cv_index(train_csv, 'train', language),
            parse_cv_index(dev_csv, 'dev', language),
            parse_cv_index(test_csv, 'test', language)
        ])


def parse_cv_index(csv_file, subset, language):
    df = pd.read_csv(csv_file)
    df['entry_id'] = df['wav_filename'].map(lambda f: splitext(basename(f))[0])
    df['subset'] = subset
    df['language'] = language
    df['start_frame'] = 0
    df['end_frame'] = df['wav_length'].map(lambda l: l // 2)  # 2 bytes per sample
    df['numeric'] = df['transcript'].map(lambda t: contains_numeric(t))
    df = df.rename(columns={'wav_filename': 'audio_file', 'wav_length': 'duration'})
    df = df.drop(['wav_filesize'], axis=1)
    return df
