from os import makedirs
from os.path import exists, basename, join
from shutil import copyfile

from tqdm import tqdm

from util.corpus_util import get_corpus

target_dir = '/media/daniel/IP9/corpora/readylingua-test'
if not exists(target_dir):
    makedirs(target_dir)

corpus = get_corpus('rl', 'de')
corpus.summary()
files = [(entry.audio_path, entry.transcript_path) for entry in set(s.entry for s in corpus.test_set())]
for audio_src, txt_src in tqdm(files):
    audio_dst = join(target_dir, basename(audio_src))
    txt_dst = join(target_dir, basename(txt_src))
    copyfile(audio_src, audio_dst)
    copyfile(txt_src, txt_dst)
