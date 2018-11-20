import re


class Vocabulary(object):

    def __init__(self, file):
        self.file = file
        self.counts_words = list(read_counts(file))

    def __len__(self):
        return len(self.counts_words)

    def __iter__(self):
        return self.words.__iter__()

    @property
    def counts(self):
        return [e[0] for e in self.counts_words]

    @property
    def words(self):
        return [e[1] for e in self.counts_words]


def read_counts(file):
    p = re.compile('(\d*)\s([a-zA-Zäöü<>]*)')
    with open(file, 'r') as f:
        for m in (p.search(line) for line in f.readlines()):
            yield m.group(1), m.group(2)
