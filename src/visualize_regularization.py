import argparse
from os.path import exists, abspath, join

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from util.visualization_util import collect_metrics, collect_loss

sns.set()

parser = argparse.ArgumentParser(description='Visualize the learning progress by plotting the learning curves for two'
                                             'models, one regularized and the other not.')
parser.add_argument('--base', type=str, help='root directory containing unregularized model')
parser.add_argument('--reg', type=str, help='root directory containing regularized model')
parser.add_argument('-t', '--target_dir', type=str, help='target directory for output')
args = parser.parse_args()


def main(args):
    dir_base, dir_reg, target_dir = setup(args)
    print(f'output will be written to {target_dir}')

    train_loss_base = collect_loss(dir_base)[('CTC_train', '1000_min')].to_frame('training')
    val_loss_base = collect_loss(dir_base)[('CTC_val', '1000_min')].to_frame('validation')
    train_loss_reg = collect_loss(dir_reg)[('CTC_train', '1000_min')].to_frame('training (regularized model)')
    val_loss_reg = collect_loss(dir_reg)[('CTC_val', '1000_min')].to_frame('validation (regularized model')
    df_loss = pd.concat([train_loss_base, train_loss_reg, val_loss_base, val_loss_reg], axis=1)

    fig, (ax_loss, ax_ler) = plt.subplots(1, 2, figsize=(10, 5))
    df_loss.plot(style=['g-', 'b-', 'g--', 'b--'], ax=ax_loss)
    ax_loss.set_xlim(1)
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.set_title('CTC loss')

    m_base = collect_metrics(dir_base)
    m_reg = collect_metrics(dir_reg)
    ler_nolm_base = m_base[('LER', 'beam', '1000_min', 'lm_n')].to_frame('LER (no spell-checking)')
    ler_lm_base = m_base[('LER', 'beam', '1000_min', 'lm_y')].to_frame('LER (with spell-checking)')
    ler_nolm_reg = m_reg[('LER', 'beam', '1000_min', 'lm_n')].to_frame('LER (regularized model, no spell-checking')
    ler_lm_reg = m_reg[('LER', 'beam', '1000_min', 'lm_y')].to_frame('LER (regularized model), with spell-checking)')
    df_metrics = pd.concat([ler_nolm_base, ler_nolm_reg, ler_lm_base, ler_lm_reg], axis=1)
    df_metrics.plot(style=['g-', 'b-', 'g--', 'b--'], ax=ax_ler)
    ax_ler.set_xlim(1)
    ax_ler.set_xlabel('epoch')
    ax_ler.set_ylabel('LER')
    ax_ler.set_title('LER')

    loss_png = join(target_dir, 'regularization.png')
    fig.tight_layout()
    fig.savefig(loss_png)
    print(f'figure saved to {loss_png}')


def setup(args):
    dir_base = abspath(args.base)
    if not exists(dir_base):
        print(f'ERROR: directory {dir_base} does not exist')
        exit(1)

    dir_reg = abspath(args.reg)
    if not exists(dir_reg):
        print(f'ERROR: directory {dir_reg} does not exist')

    target_dir = abspath(args.target_dir if args.target_dir else '.')

    return dir_base, dir_reg, target_dir


if __name__ == '__main__':
    main(args)
