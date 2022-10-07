import json
import logging
import os
import re
import tqdm
import numpy as np
import pandas as pd
from absl import app, flags
from collections import OrderedDict
from numbers import Number
from typing import List, Any, Dict, Tuple, Callable, Optional
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
flags.mark_flags_as_required(['input', 'output'])

TRUE_VALS = ['True', True]
FALSE_VALS = ['False', False]
UNDEFINED_VALS = ['Undefined', None]


def _get_perc_categorical(_df: pd.DataFrame, possible_vals: List[Any]) -> Tuple[Optional[Number], Optional[Number]]:
    _df.set_index(_df.columns[0], inplace=True)
    total = 0
    count = 0
    undef = 0
    for col in _df.T.columns:
        n = int(_df.loc[col, '0'])
        if col in possible_vals:
            count += n
        elif col in UNDEFINED_VALS:
            undef += n
            continue  # ignore undefined
        total += n
    return (None, None) if total == 0 else (int(count / total * 100), None)


def _extract_feature_stats(
        stats_dirs: Dict[str, str],
        stats_names_feats_funcs: List[
            Tuple[str, str, Callable[[pd.DataFrame, List[Any]], Tuple[Optional[Number], Optional[Number]]]]]) -> \
        Dict[str, List[str]]:
    # extracts stats for each feature of each gent according to given extract function
    stats: Dict[str, List[str]] = {}
    for name, feat, func in tqdm.tqdm(stats_names_feats_funcs):
        # logging.info(f'Processing feature "{feat}"...')
        feat_vals = []
        for ag_label, stats_dir in stats_dirs.items():
            file_path = os.path.join(stats_dir, _get_feature_file_name(feat.lower(), 'csv'))
            if not os.path.isfile(file_path):
                feat_vals.append('0')
                continue
            df = pd.read_csv(file_path)
            mean, std = func(df)
            stat = f'{mean:,.2f}' if isinstance(mean, float) else f'{mean:,}' if isinstance(mean, int) else '0'
            stat += f' ± {std:,.2f}' if isinstance(std, float) else f' ± {std:,}' if isinstance(std, int) else ''
            feat_vals.append(stat)
        stats[name] = feat_vals
    return stats


def _get_trace_stats(stats_dirs: Dict[str, str]) -> Dict[str, List[str]]:
    logging.info('Getting trace length stats comparison...')

    # loads ep length dataframes for each agent
    dfs = {ag_label: pd.read_csv(os.path.join(stats_dir, f'{EP_LENGTHS_FILE}.csv'))
           for ag_label, stats_dir in stats_dirs.items()}

    # num traces and trace length
    max_len = np.max([np.max(dfs[ag_label][EP_LENGTH_COL])
                      for ag_label in dfs.keys()])  # assumes that at least one agent reached the true max...
    return {'Num. traces': [f'{len(dfs[ag_label]):,}' for ag_label in dfs.keys()],
            'Mean trace length': [f'{dfs[ag_label][EP_LENGTH_COL].mean():,.2f} ± '
                                  f'{dfs[ag_label][EP_LENGTH_COL].std():,.2f}'
                                  for ag_label in dfs.keys()],
            'Max trace length': [f'{dfs[ag_label][EP_LENGTH_COL].max():,.0f}' for ag_label in dfs.keys()],
            '% Episodes that timed-out': [
                f'{np.sum(dfs[ag_label][EP_LENGTH_COL] == max_len) / len(dfs[ag_label]) * 100:,.0f}'
                for ag_label in dfs.keys()]}


def _get_episode_end_stats(stats_dirs: Dict[str, str]) -> Dict[str, List[str]]:
    logging.info('Getting per episode stats comparison...')
    stats_dirs = {label: os.path.join(stats_dir, LAST_STEP_DIR) for label, stats_dir in stats_dirs.items()}

    stats_names_feats_funcs = [
        ('% Episodes CC destroyed', 'Number_Enemy_CommandCenter',
         lambda _df: (int(np.sum(_df == 0, axis=0) / len(_df) * 100), None)),
        ('% Episodes agent defeated', 'Number_Friendly_Blue',
         lambda _df: (int(np.sum(_df == 0, axis=0) / len(_df) * 100), None)),
        ('Mean final Starport count', 'Number_Enemy_Starport',
         lambda _df: (np.mean(_df, axis=0)[0], np.std(_df, axis=0)[0])),
        ('Mean final sec. obj. count', 'Number_Enemy_ProductionFacility',
         lambda _df: (np.mean(_df, axis=0)[0], np.std(_df, axis=0)[0])),
        ('Mean final agent units count', 'Number_Friendly_Blue',
         lambda _df: (np.mean(_df, axis=0)[0], np.std(_df, axis=0)[0])),
        ('Mean final opponent units count', 'Number_Enemy_Red',
         lambda _df: (np.mean(_df, axis=0)[0], np.std(_df, axis=0)[0])),
    ]
    return _extract_feature_stats(stats_dirs, stats_names_feats_funcs)


def _get_per_step_stats(stats_dirs: Dict[str, str]) -> Dict[str, List[str]]:
    logging.info('Getting per step stats comparison...')
    stats_dirs = {ag_label: os.path.join(stats_dir, ALL_STEPS_DIR) for ag_label, stats_dir in stats_dirs.items()}

    stats_names_feats_funcs = [
        # ('Mean velocity towards Mobile', 'Velocity_Friendly_Blue_Mobile', lambda _df: (np.mean(_df), np.std(_df))),
        # ('Mean velocity towards CC', 'Velocity_Friendly_Blue_CommandCenter', lambda _df: (np.mean(_df), np.std(_df))),
        # ('Mean agent health loss', 'HealthDiff_Friendly_Blue', lambda _df: (np.mean(_df), np.std(_df))),
        # ('Mean opponent health loss', 'HealthDiff_Enemy_Red', lambda _df: (np.mean(_df), np.std(_df))),
        # ('Mean CC health loss', 'HealthDiff_Enemy_CommandCenter', lambda _df: (np.mean(_df), np.std(_df))),
        ('% Time advancing towards Mobile', 'Advancing_Friendly_Blue_Mobile',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
        ('% Time advancing towards CC', 'Advancing_Friendly_Blue_CommandCenter',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
        ('% Time advancing towards sec. obj.', 'Advancing_Friendly_Blue_ProductionFacility',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
        ('% Time retreating from enemy', 'Retreating_Friendly_Blue_Red',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
        ('% Time agent attacking enemy', 'Attacking_Friendly_Blue_Red',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
        ('% Time agent attacking CC', 'Attacking_Friendly_Blue_CommandCenter',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
        ('% Time enemy attacking agent', 'Attacking_Enemy_Red_Blue',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
        ('% Time NoOp', 'Noop_Friendly_Blue',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
        ('% Time Target opponent', 'Target_Blue_Red',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
        # ('% Time AttackMove', 'AttackMove_Friendly_Blue',
        #  lambda df: _get_perc_categorical(df, TRUE_VALS)),
        ('% Time MoveGrid', 'MoveGrid_Friendly_Blue',
         lambda df: _get_perc_categorical(df, TRUE_VALS)),
    ]
    return _extract_feature_stats(stats_dirs, stats_names_feats_funcs)


def main(unused_argv):
    args = flags.FLAGS

    # checks input dirs
    if args.input is None or len(args.input) == 0:
        raise ValueError(f'Empty input provided: {args.input}!')
    stats_dirs = [stats_dir for stats_dir in args.input if os.path.isdir(stats_dir)]
    if len(stats_dirs) == 0:
        raise ValueError(f'Could not find any input directories given: {args.input}!')
    if args.labels is None:
        args.labels = []  # extract labels automatically
        for i, file in enumerate(args.input):
            ag_name = re.search(r'.+_(.+Policy.*)', file)
            if ag_name is not None:
                ag_name = ag_name.group(1).replace('Policy', '')
            else:
                ag_name = f'Agent {i}'
            args.labels.append(ag_name)
    elif len(args.labels) < len(stats_dirs):
        raise ValueError(f'Insufficient labels provided for the number of input dirs: {args.labels}!')
    stats_dirs = OrderedDict(zip(args.labels, stats_dirs))

    # checks output dir and log
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'feature_compare.log'), args.verbosity)

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
