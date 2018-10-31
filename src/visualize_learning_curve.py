import argparse
from os.path import exists, abspath

import seaborn as sns

from util.visualization_util import visualize_learning_curve

sns.set()

parser = argparse.ArgumentParser()
parser.add_argument('source_dir', type=str, help='root directory containing outputs', nargs='?')
parser.add_argument('-s', '--silent', action='store_true', help='(optional) whether to suppress showing plots')
args = parser.parse_args()


def main(args):
    source_dir, silent = setup(args)
    visualize_learning_curve(source_dir, silent=silent)


def setup(args):
    if not args.source_dir:
        args.source_dir = input('Enter source directory: ')
    source_dir = abspath(args.source_dir)
    if not exists(source_dir):
        print(f'ERROR: source dir {args.source_dir} does not exist')
        exit(1)
    return source_dir, args.silent


if __name__ == '__main__':
    main(args)
