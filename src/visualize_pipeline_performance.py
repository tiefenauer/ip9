import argparse
from os.path import abspath, exists

from util.log_util import create_args_str
from util.visualization_util import visualize_pipeline_performance

parser = argparse.ArgumentParser(description="""Visualize the performance of a pipeline""")
parser.add_argument('csv_ds', type=str, nargs='?',
                    help=f'path to CSV file holding the results of a evalution of a pipeline '
                    f'using the reference model')
parser.add_argument('csv_keras', type=str, nargs='?',
                    help=f'path to CSV file holding the results of a evalution of a pipeline '
                    f'using the reference model')
parser.add_argument('-s', '--silent', action='store_true', help='(optional) whether to suppress showing plots')
args = parser.parse_args()


def main(args):
    print(create_args_str(args))
    csv_ds, csv_keras, silent = setup(args)
    visualize_pipeline_performance(csv_keras, csv_ds, silent)


def setup(args):
    if not args.csv_ds:
        args.csv_ds = input('Enter path to CSV file containing results of pipeline using the reference model: ')
    csv_ds = abspath(args.csv_ds)
    if not exists(csv_ds):
        raise ValueError(f'ERROR: path {csv_ds} does not exist')

    if not args.csv_keras:
        args.csv_keras = input('Enter path to CSV file containing results of pipeline using the simplified model: ')
        csv_keras = abspath(args.csv_keras)
    if not exists(csv_keras):
        raise ValueError(f'ERROR: path {csv_keras} does not exist')
    return csv_ds, csv_keras, args.silent


if __name__ == '__main__':
    main(args)
