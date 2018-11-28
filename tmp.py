# from os import makedirs
# from os.path import exists, join, basename
# from shutil import copyfile
#
# from tqdm import tqdm
#
# from util.corpus_util import get_corpus
#
# target_dir = '/media/daniel/IP9/corpora/librispeech-test'
# if not exists(target_dir):
#     makedirs(target_dir)
#
# corpus = get_corpus('ls')
# demo_files = [(entry.audio_path, entry.transcript_path) for entry in set(s.entry for s in corpus.test_set())]
# for src_audio, src_transcript in tqdm(demo_files):
#     dst_audio = join(target_dir, basename(src_audio))
#     dst_transcript = join(target_dir, basename(src_transcript))
#     print(dst_audio)
#     copyfile(src_audio, dst_audio)
#     print(src_transcript)
#     copyfile(src_transcript, dst_transcript)
from util.string_util import normalize

path = '/media/daniel/IP9/corpora/librispeech-test/134647.txt'
with open(path, 'r') as fin:
    txt = normalize(fin.read(), 'en')

with open(path, 'w') as fout:
    fout.write(txt)
