import json
import logging
import os
import tqdm
import numpy as np
import pandas as pd
from absl import app, flags
from collections import OrderedDict
from feature_extractor.bin.feature_stats import EP_LENGTHS_FILE, EP_LENGTH_COL, LAST_STEP_DIR, ALL_STEPS_DIR, \
    _get_feature_file_name
from feature_extractor.util.io import create_clear_dir
from feature_extractor.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Loads stats for different feature datasets and creates a CSV file comparing the results.'

FLAGS = flags.FLAGS
del FLAGS.input
del FLAGS.output
flags.DEFINE_list('input', None, 'Paths to the directories containing the feature stats.')
flags.DEFINE_list('labels', None, 'Labels for each directory provided in input.')
flags.DEFINE_string('output', None, 'Path to the directory in which to save the files with the results.')
flags.mark_flags_as_required(['input', 'labels', 'output'])

TRUE_VAL = 'True'
FALSE_VAL = 'False'
UNDEFINED_VAL = 'Undefined'


def _get_trace_stats(stats_dirs):
    logging.info('Getting trace length stats comparison...')

    # loads ep length dataframes
    dfs = {label: pd.read_csv(os.path.join(stats_dir, EP_LENGTHS_FILE))
           for label, stats_dir in stats_dirs.items()}

    # num traces and trace length
    return {'Num. traces': [f'{len(dfs[label]):,}' for label in dfs.keys()],
            'Mean trace length': [f'{dfs[label][EP_LENGTH_COL].mean():,.2f} ± '
                                  f'{dfs[label][EP_LENGTH_COL].std():,.2f}'
                                  for label in dfs.keys()],
            '% Episodes that timed-out': [
                f'{np.sum(dfs[label][EP_LENGTH_COL] == dfs[label][EP_LENGTH_COL].max()) / len(dfs[label]) * 100:,.0f}'
                for label in dfs.keys()]}


def _get_episode_end_stats(stats_dirs):
    logging.info('Getting per episode stats comparison...')
    stats_dirs = {label: os.path.join(stats_dir, LAST_STEP_DIR) for label, stats_dir in stats_dirs.items()}

    stats_names_feats = [
        ('Mean final CC count', 'Present_Enemy_CommandCenter', lambda _df: (np.mean(_df), np.std(_df))),
        ('Mean final Starport count', 'Present_Enemy_Starport', lambda _df: (np.mean(_df), np.std(_df))),
        ('Mean final sec. obj. count', 'Present_Enemy_ProductionFacility', lambda _df: (np.mean(_df), np.std(_df))),
        ('Mean final agent units count', 'Present_Friendly_Blue', lambda _df: (np.mean(_df), np.std(_df))),
        ('Mean final opponent units count', 'Present_Enemy_Red', lambda _df: (np.mean(_df), np.std(_df))),
    ]
    stats = {}
    for name, feat, func in tqdm.tqdm(stats_names_feats):
        logging.info(f'Processing feature "{feat}"...')
        feat_vals = []
        for label, stats_dir in stats_dirs.items():
            df = pd.read_csv(os.path.join(stats_dir, _get_feature_file_name(feat.lower(), 'csv')))
            mean, std = func(df)
            feat_vals.append(f'{mean.values[0]:,.2f} ± {std.values[0]:,.2f}')
        stats[name] = feat_vals

    return stats


def _get_per_step_stats(stats_dirs):
    logging.info('Getting per step stats comparison...')
    stats_dirs = {label: os.path.join(stats_dir, ALL_STEPS_DIR) for label, stats_dir in stats_dirs.items()}

    def _get_perc_categorical(_df, possible_vals, default=0):
        _df.set_index(_df.columns[0], inplace=True)
        for possible_val in possible_vals:
            if possible_val in _df.T.columns:
                return int(_df.T[possible_val][0]), 0
        return default, default

    stats_names_feats = [
        ('Mean velocity towards Mobile', 'Velocity_Friendly_Blue_Mobile', lambda _df: (np.mean(_df), np.std(_df))),
        ('Mean velocity towards CC', 'Velocity_Friendly_Blue_CommandCenter', lambda _df: (np.mean(_df), np.std(_df))),
        ('Mean agent health loss', 'UnderAttack_Friendly_Blue', lambda _df: (np.mean(_df), np.std(_df))),
        ('Mean opponent health loss', 'UnderAttack_Enemy_Red', lambda _df: (np.mean(_df), np.std(_df))),
        ('Mean CC health loss', 'UnderAttack_Enemy_CommandCenter', lambda _df: (np.mean(_df), np.std(_df))),
        ('NoOp count', 'Noop_Friendly_Blue', lambda _df: _get_perc_categorical(df, [TRUE_VAL, True], 0)),
        ('Target opponent count', 'Target_Blue_Red', lambda _df: _get_perc_categorical(df, [TRUE_VAL, True], 0)),
        ('AttackMove count', 'AttackMove_Friendly_Blue', lambda _df: _get_perc_categorical(df, [TRUE_VAL, True], 0)),
        ('MoveGrid count', 'MoveGrid_Friendly_Blue', lambda _df: _get_perc_categorical(df, [TRUE_VAL, True], 0)),
    ]
    stats = {}
    for name, feat, func in tqdm.tqdm(stats_names_feats):
        logging.info(f'Processing feature "{feat}"...')
        feat_vals = []
        for label, stats_dir in stats_dirs.items():
            df = pd.read_csv(os.path.join(stats_dir, _get_feature_file_name(feat.lower(), 'csv')))
            mean, std = func(df)
            if isinstance(mean, pd.Series):
                feat_vals.append(f'{mean.values[0]:,.2f} ± {std.values[0]:,.2f}')
            else:
                feat_vals.append(f'{mean:,}')
        stats[name] = feat_vals

    return stats


def main(unused_argv):
    args = flags.FLAGS

    # checks input dirs
    if args.input is None or len(args.input) == 0:
        raise ValueError(f'Empty input provided: {args.input}!')
    stats_dirs = [stats_dir for stats_dir in args.input if os.path.isdir(stats_dir)]
    if len(stats_dirs) == 0:
        raise ValueError(f'Could not find any input directories given: {args.input}!')
    if len(args.labels) < len(stats_dirs):
        raise ValueError(f'Insufficient labels provided for the number of input dirs: {args.labels}!')
    stats_dirs = OrderedDict(zip(args.labels, stats_dirs))

    # checks output dir and log
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'feature_stats.log'), args.verbosity)

    # save args
    with open(os.path.join(args.output, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    logging.info(f'Processing feature stats for {len(stats_dirs.keys())} partitions: {list(stats_dirs.keys())}')

    stats = _get_trace_stats(stats_dirs)
    stats.update(_get_episode_end_stats(stats_dirs))
    stats.update(_get_per_step_stats(stats_dirs))

    df = pd.DataFrame.from_dict(stats, columns=list(stats_dirs.keys()), orient='index')
    file_name = os.path.join(args.output, 'agent_comparison.csv')
    logging.info(f'Saving stats comparison file to: {file_name}...')
    df.to_csv(file_name, encoding='utf8')
    logging.info('Done!')


if __name__ == '__main__':
    app.run(main)
