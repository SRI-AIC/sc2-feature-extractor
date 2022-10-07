import argparse
import logging
import os
import pandas as pd
from typing import List
from feature_extractor import REPLAY_FILE_STR, TIME_STEP_STR, EPISODE_STR
from feature_extractor.util.cmd_line import str2bool, save_args
from feature_extractor.util.io import create_clear_dir, get_file_name_without_extension
from feature_extractor.util.logging import change_log_handler
from feature_extractor.util.mp import run_parallel

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Generates subtitles from feature files that can be visualized during replay videos playback.'

FILE_EXTENSION = 'sub'


def _generate_subs(df: pd.DataFrame, output_file: str, features: List[str]):
    with open(output_file, 'w') as fp:
        for _, row in df.iterrows():
            start = row[TIME_STEP_STR]
            end = start + 1
            feats = []
            for feat in features:
                feat_val = row[feat]
                if isinstance(feat_val, float):
                    feats.append(f'{feat}={feat_val:.2f}')
                else:
                    feats.append(f'{feat}={feat_val}')
            feat_str = '\\n'.join(feats)
            fp.write(f'{{{start}}}{{{end}}}{feat_str}\n')


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Pickle file containing the high-level features.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory in which to save the subtitle files')
    parser.add_argument('--features', '-f', type=str, nargs='+', default=None,
                        help='The list of features to write as subtitles. `None` will write all features to file.')

    parser.add_argument('--processes', '-p', type=int, default=-1,
                        help='The number of parallel processes to use. A value of `-1` or `None` '
                             'will use all available CPUs.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise ValueError(f'Could not find features file in {args.input}')

    # check output dir and log file
    out_dir = args.output
    create_clear_dir(out_dir, args.clear)
    save_args(args, os.path.join(out_dir, 'args.json'))
    change_log_handler(os.path.join(out_dir, 'subtitles.log'), args.verbosity)

    # loads features
    logging.info('_________________________________________')
    logging.info(f'Loading features dataset from: {args.input}...')
    features_df: pd.DataFrame = pd.read_pickle(args.input)
    features_df.reset_index(drop=True, inplace=True)  # resets index in case it's timestep indexed
    logging.info(f'Loaded data for {len(features_df.columns[3:])} features, '
                 f'{len(features_df[REPLAY_FILE_STR].unique())} episodes')

    features = args.features
    if features is None:
        features = [f for f in features_df.columns if f not in {REPLAY_FILE_STR, TIME_STEP_STR, EPISODE_STR}]
    fn_args = [(df, os.path.join(out_dir, f'{get_file_name_without_extension(file)}.{FILE_EXTENSION}'), features)
               for file, df in features_df.groupby(REPLAY_FILE_STR)]
    run_parallel(_generate_subs, fn_args, args.processes, use_tqdm=True)

    logging.info('Done!')


if __name__ == '__main__':
    main()
