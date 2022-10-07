import logging
import os
import shutil
import tempfile
import pandas as pd
import tqdm
import numpy as np
from enum import IntEnum
from typing import List, Dict
from collections import OrderedDict
from absl import app, flags
from pysc2.lib import point_flag
from feature_extractor.config import FeatureExtractorConfig
from feature_extractor.util.logging import change_log_handler
from feature_extractor.util.io import create_clear_dir, save_dict_json
from feature_extractor.visualization.location_processor import ALL_GROUP
from feature_extractor.visualization.location_visualizer import LocationVisualizer

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__desc__ = 'Replays one or more SC2 games and creates visualizations based on units\' locations.'

FLAGS = flags.FLAGS
point_flag.DEFINE_point('feature_screen_size', 84, 'Resolution for screen feature layers.')
point_flag.DEFINE_point('feature_minimap_size', 64, 'Resolution for minimap feature layers.')
flags.DEFINE_integer('feature_camera_width', 24, 'Width of the feature layer camera.')
flags.DEFINE_string('action_space', 'RAW', 'Action space for agent interface format.')

flags.DEFINE_string('config', None, 'Path to the feature extractor configuration file.')
flags.DEFINE_string('output', None, 'Path to the directory in which to save the results.')
flags.DEFINE_bool('clear', False, 'Whether to clear output directories before generating results.')
flags.DEFINE_bool('dark', True, 'Whether to use a dark theme/background for plotted figures.')
flags.DEFINE_string('clusters_file', None, 'The path to the clusters CSV file to be used, containing a reference for '
                                           'the replay file of each trace and corresponding cluster. Results are going'
                                           'to be processed per cluster. ')
flags.DEFINE_string('format', 'png', 'Image format for plotted figures.')
flags.mark_flags_as_required(['output', 'config'])

CLUSTER_ID_COL = 'Cluster'
REPLAY_FILE_STR = 'Trace ID'


def _convert_unit_filter(config: FeatureExtractorConfig, unit_filter: List[str or IntEnum]) -> Dict[str, np.ndarray]:
    """
    Converts a filter into a dictionary of sets of units to facilitate feature extraction.
    :param FeatureExtractorConfig config: the feature configuration containing the unit groups definitions.
    :param list[str or IntEnum] unit_filter: the unit filter for an extractor.
    :rtype: OrderedDict[str, np.ndarray]
    :return: a dictionary of sets of units for which we want to separate feature extraction.
    """
    groups = OrderedDict([(g, np.array([g_unit.value for g_unit in config.groups[g]]))
                          if g in config.groups else (g.name, np.array([g.value]))
                          for g in unit_filter])
    if ALL_GROUP not in groups:
        groups[ALL_GROUP] = np.unique(np.concatenate(list(groups.values())))  # create group with all units
    return groups


def main(unused_argv):
    args = FLAGS

    # try to load config, save it in output dir and load unit group filters
    if not os.path.isfile(args.config):
        raise ValueError(f'Configuration file does not exist: {args.config}')

    # checks output dir and files
    create_clear_dir(args.output, args.clear)
    change_log_handler(os.path.join(args.output, 'loc-visualizer.log'), args.verbosity)
    save_dict_json({a: args[a].value if hasattr(args[a], 'value') else str(args[a]) for a in args},
                   os.path.join(args.output, 'args.json'))

    config = FeatureExtractorConfig.load_json(args.config)
    config.save_json(os.path.join(args.output, os.path.basename(args.config)))
    friendly_groups = _convert_unit_filter(config, config.unit_group_friendly_filter)
    enemy_groups = _convert_unit_filter(config, config.unit_group_enemy_filter)

    # if clusters mode, loads zip files from directory containing CSV with traces data (and path to replay files)
    if args.clusters_file is not None:
        assert os.path.isfile(args.clusters_file), f'Cannot find clusters file in: {args.clusters_file}'
        replays_df = pd.read_csv(args.clusters_file)
        logging.info(f'Using clusters mode, loaded {len(replays_df)} traces files from: {args.clusters_file}...')

        replays = {}
        for cluster, cluster_df in tqdm.tqdm(replays_df.groupby(CLUSTER_ID_COL)):
            # get replay files
            replay_files = cluster_df[REPLAY_FILE_STR].unique()

            # copy replays to temp directory
            replay_dir = tempfile.TemporaryDirectory().name
            os.makedirs(replay_dir, exist_ok=True)
            output_dir = os.path.join(args.output, f'cluster-{cluster}')
            create_clear_dir(output_dir, args.clear)
            replays[replay_dir] = output_dir
            for replay_file in replay_files:
                replay_file = os.path.join(args.replays, f'{replay_file}')
                if not replay_file.endswith('.SC2Replay'):
                    replay_file += '.SC2Replay'  # add extension if needed
                if os.path.isfile(replay_file):
                    shutil.copy(replay_file, replay_dir)
    else:
        replays = {args.replays: args.output}

    # create visualizer
    loc_visualizer = LocationVisualizer(
        args.feature_screen_size, args.feature_minimap_size, args.action_space, args.feature_camera_width,
        True, True, True, args.verbosity, args.parallel, args.dark, args.format)

    # process replays and saves results
    for replay_dir, output_dir in replays.items():
        loc_visualizer.visualize_replays(
            replay_dir, friendly_groups, enemy_groups, output_dir, args.replay_sc2_version)

    # cleanup
    if args.clusters_file is not None:
        logging.info('Cleaning up temporary replay directories..')
        for replay_dir in tqdm.tqdm(replays.keys()):
            shutil.rmtree(replay_dir)

    logging.info('Done!')


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == '__main__':
    app.run(main)
