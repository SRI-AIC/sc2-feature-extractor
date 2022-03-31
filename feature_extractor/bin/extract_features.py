import json
import logging
import os
import shutil
import sys
import time
from typing import Dict, List
from absl import app, flags
from feature_extractor.extractors import MetaExtractor, FeatureExtractor, FRIENDLY_STR, ENEMY_STR
from feature_extractor.extractors.orders import FriendlyOrdersExtractor, EnemyOrdersExtractor
from feature_extractor.extractors.factors.force import ForceFactorsExtractor
from feature_extractor.extractors.factors.force_relative import ForceRelativeFactorsExtractor
from feature_extractor.extractors.factors.under_attack import UnderAttackExtractor
from feature_extractor.extractors.group import UnitGroupExtractor
from feature_extractor.extractors.location.between import BetweenExtractor
from feature_extractor.extractors.location.concentration import ConcentrationExtractor
from feature_extractor.extractors.location.distance import DistanceExtractor
from feature_extractor.extractors.location.elevation import ElevationExtractor
from feature_extractor.extractors.location.movement import FriendlyRelativeMovementExtractor, \
    EnemyRelativeMovementExtractor
from pysc2.lib import features, point_flag
from feature_extractor import merge_feature_files, REPLAY_FILE_STR
from feature_extractor.config import FeatureExtractorConfig
from feature_extractor.extractor import ExtractorProcessor, ExtractorListener
from feature_extractor.replayer import ReplayProcessRunner
from feature_extractor.util.data import save_separate_csv_gzip
from feature_extractor.util.io import create_clear_dir, get_files_with_extension
from feature_extractor.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Run SC2 to replay a game and extract high-level features. '

# flags
FLAGS = flags.FLAGS
point_flag.DEFINE_point('feature_screen_size', 64, 'Resolution for screen feature layers.')
point_flag.DEFINE_point('feature_minimap_size', 64, 'Resolution for minimap feature layers.')
flags.DEFINE_integer('feature_camera_width', 24, 'Width of the feature layer camera.')
flags.DEFINE_string('action_space', 'RAW', 'Action space for agent interface format.')

flags.DEFINE_bool('categorical', True, 'Whether to extract categorical features. If `False`, then'
                                       'continuous/numerical features will be computed where possible.')
flags.DEFINE_string('config', None, 'Path to the feature extractor configuration file.')
flags.DEFINE_integer('amount', None, 'Number of replays for which to extract features (selected '
                                     'from the list of replays).')
flags.DEFINE_string('output', 'output',
                    'Path to the directory in which to save the file with the extracted features.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results')
flags.DEFINE_bool('keep_csv', True,
                  'Whether to keep he individual CSV feature files after generating the compressed files')

flags.mark_flags_as_required(['replays', 'config'])

PICKLE_DATASET_FILE = 'feature-dataset.pkl.gz'
SEPARATE_DATASET_FILE = 'all-traces.tar.gz'


def _create_extractors(meta_extractor: MetaExtractor,
                       categorical: bool) -> Dict[str, List[FeatureExtractor]]:
    config = meta_extractor.config
    return {
        FRIENDLY_STR: [
            meta_extractor,
            UnitGroupExtractor(config, categorical),
            DistanceExtractor(config, categorical),
            ConcentrationExtractor(config, categorical),
            ForceFactorsExtractor(config, categorical),
            ForceRelativeFactorsExtractor(config, categorical),
            UnderAttackExtractor(config, categorical),
            ElevationExtractor(config, categorical),
            FriendlyRelativeMovementExtractor(config, categorical),
            EnemyRelativeMovementExtractor(config, categorical),
            BetweenExtractor(config, categorical),
            FriendlyOrdersExtractor(config),
        ],
        ENEMY_STR: [
            EnemyOrdersExtractor(config)
        ]}


def main(unused_argv):
    FLAGS(sys.argv)
    args = flags.FLAGS

    # check config
    if not os.path.exists(args.config):
        raise ValueError(f'Config file does not exist: {args.config}.')

    # checks output dir and files
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'extractor.log'), args.verbosity)

    # save args
    with open(os.path.join(args.output, 'args.json'), 'w') as fp:
        json.dump({k: args[k].value for k in args}, fp, indent=4)

    # load config, save to output dir
    config = FeatureExtractorConfig.load_json(FLAGS.config)
    config.save_json(os.path.join(FLAGS.output, os.path.basename(FLAGS.config)))

    # creates agent interface format and sample converter
    aif = features.parse_agent_interface_format(
        camera_width_world_units=args.feature_camera_width,
        use_camera_position=True,
        use_feature_units=True,
        use_raw_units=True,
        action_space=args.action_space,
        feature_screen=args.feature_screen_size,
        feature_minimap=args.feature_minimap_size
    )

    # creates and runs the replay processor
    temp_dir = os.path.join(args.output, 'ep_data')
    create_clear_dir(temp_dir, args.clear)
    time.sleep(1)  # not sure why but sometimes temp dir is not created

    # creates feature extractors
    meta_extractor = MetaExtractor(config)
    extractors = _create_extractors(meta_extractor, args.categorical)

    listener = ExtractorListener(meta_extractor, extractors, temp_dir)
    extractor = ExtractorProcessor(listener, aif)
    if args.parallel == -1:
        args.parallel = os.cpu_count()
        logging.info(f'Using all cpus available ({args.parallel})')
    runner = ReplayProcessRunner(args.replays, extractor, args.replay_sc2_version,
                                 args.parallel, args.episodes, amount=args.amount)
    runner.run()

    # gathers all files
    files = get_files_with_extension(temp_dir, 'csv')
    logging.info('=========================================')
    logging.info(f'Found {len(files)} CSV feature files, merging into single dataset... ')
    df = merge_feature_files(files, dtype=str, show_progress=True, use_replay_name=True)

    # saves to single pickle file
    file_path = os.path.join(args.output, PICKLE_DATASET_FILE)
    logging.info(f'Saving file with features for all episodes to:\n\t{file_path}...')
    df.to_pickle(file_path, compression='gzip')

    # saves also to separate csv files inside single gzip file
    file_path = os.path.join(args.output, SEPARATE_DATASET_FILE)
    save_separate_csv_gzip(df, file_path, group_by=REPLAY_FILE_STR, use_group_filename=True)

    # removes CSV files if requested
    if not args.keep_csv:
        shutil.rmtree(temp_dir)

    logging.info('Done!')


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == '__main__':
    app.run(main)
