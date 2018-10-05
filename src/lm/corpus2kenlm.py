import argparse
import multiprocessing as mp

from lm.lm_preprocessing import process_line
from tqdm import tqdm

from util.string_util import remove_multi_spaces, unidecode_keep_umlauts, remove_punctuation, replace_numeric


def main(input_path, output_path, num_lines=None, num_threads=1):
    print(f'processing {num_lines} from {input_path} to {output_path} using {num_threads} workers')
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out, mp.Pool(num_threads) as pool:
        # process non-empty lines that don't start with tabulator (often literature and links, not sentences)
        lines = (line for line in f_in if not line.startswith('\t') and not line.startswith('\n'))
        processed_lines = tqdm(pool.imap(process_line, lines, chunksize=32), total=num_lines, unit=' lines')
        for sentence in (sentence for processed_line in processed_lines for sentence in processed_line):
            f_out.write(sentence)


def normalize_sentence(sentence):
    return remove_multi_spaces(
        unidecode_keep_umlauts(remove_punctuation(replace_numeric(sentence, by_single_digit=True))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Tokenize text corpus to KenLM compatible input file (1 sentence per line)""")
    parser.add_argument('--input', type=str, help='path to input file to read from')
    parser.add_argument('--output', type=str, help='path to output file to read to')
    parser.add_argument('--num_lines', type=int, nargs='?', default=None,
                        help='(optional) number of lines in the input file. If set a progress bar will be shown.')
    parser.add_argument('--threads', type=int, nargs='?', default=mp.cpu_count(),
                        help='(optional) number of threads (for multiprocessing)')
    args = parser.parse_args()
    main(args.input, args.output, args.num_lines, args.threads)
