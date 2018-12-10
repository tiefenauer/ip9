import argparse
from os.path import exists, abspath, join

import seaborn as sns
from matplotlib import pyplot as plt

from util.visualization_util import collect_metrics, plot_df

sns.set()

parser = argparse.ArgumentParser(description='Visualize the learning progress by plotting the learning curves for two'
                                             'models, one regularized and the other not.')
parser.add_argument('--base', type=str, help='root directory containing unregularized model')
parser.add_argument('--reg', type=str, help='root directory containing regularized model')
parser.add_argument('--synth', type=str, help='root directory containing unregularized model (synthesized data)')
parser.add_argument('--reg_synth', type=str, help='root directory containing regularized model (synthesized data)')
parser.add_argument('-t', '--target_dir', type=str, help='target directory for output')
args = parser.parse_args()


def main(args):
    dir_base, dir_reg, dir_synth, dir_reg_synth, target_dir = setup(args)
    print(f'output will be written to {target_dir}')

    ler_base = collect_metrics(dir_base)[('LER', 'beam', '1000_min', 'lm_n')].to_frame()
    ler_reg = collect_metrics(dir_reg)[('LER', 'beam', '1000_min', 'lm_n')].to_frame()
    ler_synth = collect_metrics(dir_synth)[('LER', 'beam', '1000_min', 'lm_n')].to_frame()
    ler_reg_synth = collect_metrics(dir_reg_synth)[('LER', 'beam', '1000_min', 'lm_n')].to_frame()

    df_reg = ler_base.join(ler_reg, lsuffix='_base', rsuffix='_reg')
    df_synth = ler_base.join(ler_synth, lsuffix='_base', rsuffix='_synth')
    df_reg_synth = ler_base.join(ler_reg_synth, lsuffix='_base', rsuffix='_reg_synth')

    fig, (ax_reg, ax_synth, ax_reg_synth) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    styles = ['g-', 'b-']
    plot_df(df_reg, ax_reg, styles, ['unregularized model', 'regularized model'], 'LER', 'Regularisation')
    plot_df(df_synth, ax_synth, styles, ['no synthesized data', 'with synthesized data'], 'LER', 'Synthetization')
    plot_df(df_reg_synth, ax_reg_synth, styles,
            ['unregularized model, no synthesized data', 'regularized model, synthesized data'],
            'LER', 'Synthetisation & Regularisation')

    regularization_synthetisation_png = join(target_dir, 'regularization_synthetisation.png')
    fig.tight_layout()
    fig.savefig(regularization_synthetisation_png)
    print(f'loss plot saved to {regularization_synthetisation_png}')


def setup(args):
    dir_base = abspath(args.base)
    if not exists(dir_base):
        print(f'ERROR: directory {dir_base} does not exist')
        exit(1)

    dir_reg = abspath(args.reg)
    if not exists(dir_reg):
        print(f'ERROR: directory {dir_reg} does not exist')

    dir_synth = abspath(args.synth)
    if not exists(dir_synth):
        print(f'ERROR: directory {dir_synth} does not exist')

    dir_reg_synth = abspath(args.reg_synth)
    if not exists(dir_reg_synth):
        print(f'ERROR: directory {dir_reg_synth} does not exist')

    target_dir = abspath(args.target_dir if args.target_dir else '.')

    return dir_base, dir_reg, dir_synth, dir_reg_synth, target_dir


if __name__ == '__main__':
    main(args)
