import json
import logging
import os
import re
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Union, Dict, Optional
from absl import app, flags
from pandas.core.groupby import DataFrameGroupBy
from feature_extractor import merge_feature_files
from feature_extractor.extractors import TIME_STEP_STR, EPISODE_STR, REPLAY_FILE_STR
from feature_extractor.util.logging import change_log_handler
from feature_extractor.util.mp import run_parallel
from feature_extractor.util.plot import plot_bar, dummy_plotly, plot_histogram
from feature_extractor.util.io import get_files_with_extension, create_clear_dir, get_file_changed_extension

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Get stats from one or more feature CSV files and saves results for each feature to a file.'

FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'Path to the file (zip or CSV) or directory containing the extracted features.')
flags.DEFINE_string('output', None, 'Path to the directory in which to save the files with the results.')
flags.DEFINE_string('desc', None, 'Path to the JSON file containing the feature descriptions.')
flags.DEFINE_string('match', None, 'Regex used to filter features by name.')
flags.DEFINE_string('format', 'png', 'File format of images with the resulting plots.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directory before generating results.')
flags.DEFINE_integer("parallel", 1, "How many processes to run in parallel.")
flags.mark_flags_as_required(['input', 'output'])

FIRST_STEP_DIR = 'first-step'
LAST_STEP_DIR = 'last-step'
ALL_STEPS_DIR = 'all-steps'
ALL_EPISODES_DIR = 'all-episodes'
SEQUENCE_DIR = 'sequence'

DESCRIPTIVE_STATS_FILE = 'descriptive_stats.csv'
EP_LENGTHS_FILE = 'ep_lengths'
EP_LENGTH_COL = 'Length'

HIST_PALETTE = 'Portland'


def _get_feature_file_name(feature: str, ext: str) -> str:
    feature = feature.lower().replace('"', '')
    return f'{feature}.{ext}'


def _plot_feature_bar(feature: str, df_counts: Union[pd.DataFrame, pd.Series], out_dir: str, img_format: str):
    if len(df_counts) == 0:
        logging.info(f'No data found for feature {feature}, skipping')
    if isinstance(df_counts, pd.Series):
        df_counts = df_counts.to_dict()
    file_name = os.path.join(out_dir, _get_feature_file_name(feature, img_format))
    plot_bar(df_counts, feature, file_name, x_label=' ', y_label='Count', plot_mean=True)


def _plot_feature_histogram(feature: str, df_values: Union[pd.DataFrame, pd.Series],
                            val_range: Tuple[float, float], out_dir: str, img_format: str):
    if len(df_values) == 0:
        logging.info(f'No data found for feature {feature}, skipping')
    if isinstance(df_values, pd.Series):
        df_values = df_values.to_frame()
    file_name = os.path.join(out_dir, _get_feature_file_name(feature, img_format))

    def _get_range_val(idx):
        if val_range is None:
            return None
        val = val_range[idx]
        if val == sys.maxsize or val == np.finfo(np.float).max or val == np.finfo(np.float).min:
            return None
        return val

    plot_histogram(df_values, f'{feature} Histogram', file_name, x_label=' ', y_label='Count',
                   plot_mean=True, show_legend=False, palette=HIST_PALETTE)
    # x_min=_get_range_val(0), x_max=_get_range_val(1))


def _plot_feature_stats(feature: str,
                        df: pd.DataFrame,
                        first_step_df: pd.DataFrame,
                        by_ep: DataFrameGroupBy,
                        feature_range: Optional[Tuple[float, float]],
                        first_dir: str,
                        last_dir: str,
                        all_steps_dir: str,
                        all_eps_dir: str,
                        sequence_dir: str,
                        img_format: str):
    dummy_plotly()  # just to get rid of weird messages in plotly plots

    logging.info(f'Processing feature {feature}...')
    by_ep_feat = by_ep[feature]
    if df[feature].dtype == object:
        feat_vals = df[feature].unique()
        _plot_feature_bar(feature, first_step_df.groupby(feature).size(), first_dir, img_format)
        _plot_feature_bar(feature, df.groupby(feature).size(), all_steps_dir, img_format)

        counts = pd.Series({f: np.mean([(g == f).cumsum().max() for e, g in by_ep_feat]) for f in feat_vals},
                           name=feature)
        _plot_feature_bar(feature, counts, sequence_dir, img_format)

        counts = pd.Series({f: np.sum([np.any(g == f) for e, g in by_ep_feat]) for f in feat_vals},
                           name=feature)
        _plot_feature_bar(feature, counts, all_eps_dir, img_format)

        counts = pd.Series({f: np.sum([g.values[-1] == f for e, g in by_ep_feat]) for f in feat_vals},
                           name=feature)
        _plot_feature_bar(feature, counts, last_dir, img_format)

    else:
        _plot_feature_histogram(feature, first_step_df[feature], feature_range, first_dir, img_format)
        _plot_feature_histogram(feature, df[feature], feature_range, all_steps_dir, img_format)

        _plot_feature_histogram(
            feature, pd.Series([np.mean(g.values) for e, g in by_ep_feat], name=feature),
            feature_range, all_eps_dir, img_format)

        _plot_feature_histogram(
            feature, pd.Series([g.values[-1] for e, g in by_ep_feat], name=feature),
            feature_range, last_dir, img_format)


def main(unused_argv):
    args = flags.FLAGS

    # checks input files
    files = []
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = list(get_files_with_extension(args.input, 'csv')) + \
                list(get_files_with_extension(args.input, 'tar.gz')) + \
                list(get_files_with_extension(args.input, 'pkl.gz'))
    if len(files) == 0:
        raise ValueError(f'Input path is not a valid file or directory: {args.input}!')

    # checks output dirs and files
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'feature_stats.log'), args.verbosity)

    first_dir = os.path.join(args.output, FIRST_STEP_DIR)
    create_clear_dir(first_dir)
    with open(os.path.join(first_dir, 'README.txt'), 'w') as fp:
        fp.write('Statistics regarding first step values of features across replays: counts for each category '
                 '(categorical mode) / distribution (numeric mode)')

    last_dir = os.path.join(args.output, LAST_STEP_DIR)
    create_clear_dir(last_dir)
    with open(os.path.join(last_dir, 'README.txt'), 'w') as fp:
        fp.write('Statistics regarding last step values of features across replays: counts for each category '
                 '(categorical mode) / distribution (numeric mode)')

    all_steps_dir = os.path.join(args.output, ALL_STEPS_DIR)
    create_clear_dir(all_steps_dir)
    with open(os.path.join(all_steps_dir, 'README.txt'), 'w') as fp:
        fp.write('Statistics regarding features\' values across all timesteps of replays: total counts for each '
                 'category (categorical mode) / distribution (numeric mode)')

    all_eps_dir = os.path.join(args.output, ALL_EPISODES_DIR)
    create_clear_dir(all_eps_dir)
    with open(os.path.join(all_eps_dir, 'README.txt'), 'w') as fp:
        fp.write('Statistics regarding features\' values over all replays: number of episodes in which a '
                 'category was present in at least one timestep (categorical mode) / mean value distribution '
                 'per episode (numeric mode)')

    sequence_dir = os.path.join(args.output, SEQUENCE_DIR)
    create_clear_dir(sequence_dir)
    with open(os.path.join(sequence_dir, 'README.txt'), 'w') as fp:
        fp.write('Mean of maximum consecutive constant feature value steps per episode (categorical mode only)')

    # save args
    with open(os.path.join(args.output, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # merge files into single data frame
    df = merge_feature_files(files, dtype=str, use_replay_name=False)
    df = df.iloc[1:]  # ignore first lines as they have invalid values

    # tries to load feature descriptors file
    features_ranges: Dict[str, Tuple[int, int]] = {}
    if args.desc is not None and os.path.isfile(args.desc):
        with open(args.desc, 'r') as fp:
            desc = json.load(fp)
        features_ranges = {feat_desc['name']: feat_desc['values'] for feat_desc in desc['conditions']}
        features_ranges.update({feat_desc['name']: feat_desc['values']
                                for _, feat_descs in desc['tactics'].items()
                                for feat_desc in feat_descs})
        logging.info(f'Loaded descriptions for {len(features_ranges)} features from "{args.desc}"')
    else:
        logging.info(f'Could not load feature descriptions from "{args.desc}"')

    dummy_plotly()  # just to get rid of weird messages in plotly plots

    # trace stats
    by_ep = df.groupby(EPISODE_STR)
    length_df = pd.DataFrame([len(ep_df) for ep, ep_df in by_ep], columns=[EP_LENGTH_COL])
    logging.info(f'Got {len(length_df)} episodes, mean trace length of '
                 f'{length_df["Length"].mean():.2f}±{length_df["Length"].std():.2f}')
    file_name = os.path.join(args.output, f'{EP_LENGTHS_FILE}.{args.format}')
    plot_histogram(length_df, 'Episode Length Histogram', file_name,
                   x_label=' ', y_label='Count',
                   plot_mean=True, show_legend=False, palette=HIST_PALETTE)
    logging.info(f'Saved episode lengths to:\n\t{file_name}')

    # descriptive stats
    file_name = os.path.join(args.output, DESCRIPTIVE_STATS_FILE)
    df.describe(include='all').to_csv(file_name)
    logging.info(f'Saved descriptive stats for {len(length_df)} episodes to:\n\t{file_name}')

    # per feature stats
    df[TIME_STEP_STR] = pd.to_numeric(df[TIME_STEP_STR])
    first_step_df = df[df[TIME_STEP_STR] == 1]  # selects only first step of episodes
    first_feat_idx = df.columns.values.tolist().index(REPLAY_FILE_STR) + 1

    # add function arguments
    feat_args = []
    for feature in df.columns[first_feat_idx:]:
        if args.match is None or re.search(args.match, feature):
            feat_args.append((feature, df, first_step_df, by_ep,
                              features_ranges[feature] if feature in features_ranges else None,
                              first_dir, last_dir, all_steps_dir, all_eps_dir, sequence_dir, args.format))

    # processes features in parallel
    logging.info(f'Extracting feature stats for {len(feat_args)} features...')
    run_parallel(_plot_feature_stats, feat_args, args.parallel, use_tqdm=True)
    logging.info(f'Finished processing {len(feat_args)} features ({len(df[EPISODE_STR].unique())} episodes)!')


if __name__ == '__main__':
    app.run(main)
