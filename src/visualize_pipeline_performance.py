import argparse
from os.path import abspath, exists

from util.log_util import create_args_str
from util.pipeline_util import visualize_performance

parser = argparse.ArgumentParser(description="""Visualize the performance of a pipelins""")
parser.add_argument('csv', type=str, nargs='?',
                    help=f'path to CSV file holding the results of a pipeline evaluation')
parser.add_argument('-s', '--silent', action='store_true', help='(optional) whether to suppress showing plots')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))
    df_path, silent = setup(args)
    visualize_performance(df_path, silent)


def setup(args):
    if not args.csv:
        args.csv = input('Enter path to CSV file to visualize: ')
    csv_path = abspath(args.csv)
    if not exists(csv_path):
        raise ValueError(f'ERROR: path {csv_path} does not exist')
    return csv_path, args.silent


if __name__ == '__main__':
    main(args)
