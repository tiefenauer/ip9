import sys
from datetime import datetime
from os.path import join


def get_num_features(feature_type):
    if feature_type == 'pow':
        return 161
    elif feature_type == 'mel':
        return 40
    elif feature_type == 'mfcc':
        return 13
    print(f'error: unknown feature type: {feature_type}', file=sys.stderr)
    exit(1)


def get_target_dir(infix, args):
    target_dir = join(args.target_root, datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + infix)
    target_dir += '_' + args.corpus
    target_dir += '_' + args.language
    # target_dir += '_' + args.feature_type
    return target_dir
