import gc
import io
import logging
import os
import multiprocessing as mp
import queue
import time
import skvideo.io
import tqdm
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Union, Optional, List, Any
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from pysc2.lib import features
from pysc2.lib.features import PlayerRelative
from feature_extractor.replayer import ReplayProcessRunner
from feature_extractor.util.io import save_object, load_object
from feature_extractor.util.mp import run_parallel
from feature_extractor.visualization.location_processor import LocationTrackingProcessor, ALL_GROUP

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

SELF = PlayerRelative.SELF
ENEMY = PlayerRelative.ENEMY
NEUTRAL = PlayerRelative.NEUTRAL

TIMEOUT = 5  # max secs to wait for results queue

FRIENDLY_COLOR_MAP = plt.cm.winter_r
ENEMY_COLOR_MAP = plt.cm.autumn_r
NEUTRAL_CMAP = plt.cm.Greys
NEUTRAL_CMAP_DARK = plt.cm.Greys_r

INTERSECTION_ALPHA = 0.6  # to blend where there's friendly and enemy units
ALPHA_EXP_FACTOR = 10  # 40  # 10  # transparency factor for histograms (the higher the less importance the relative counts have)

DPI = 600  # rasterized image dots per inch (for still images)
ANIM_DURATION = 6  # animation duration in seconds
ANIM_FPS = 20  # number of images / timesteps to be saved in the  animation per second
NUM_FRAMES = ANIM_FPS * ANIM_DURATION

SideGroupsHistogramList = Dict[PlayerRelative, Dict[str, List]]


def gradient_line_legend(color_maps, labels, num_points=10, handle_length=3):
    """
    Creates a legend where each entry is a gradient color line.
    :param list color_maps: the color maps used in the legend.
    :param list[str] labels: the labels of the legend entries.
    :param int num_points: the number of points used to create the gradient.
    :param int handle_length: the length of the legend line entries.
    """
    assert len(color_maps) == len(labels), 'Number of color maps has to be the same as that of labels!'
    color_space = np.linspace(0, 1, num_points)
    lines = []
    for c_map in color_maps:
        lines.append(tuple(Line2D([], [], marker='s', markersize=handle_length, c=c_map(c)) for c in color_space))

    plt.legend(lines, labels, numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)},
               handlelength=handle_length)


class LocationVisualizer(object):
    """
    A visualizer for SC2 replays that plots the locations of groups of units over time for friendly and enemy forces
    on a 2D grid map.
    """

    def __init__(self,
                 feature_screen: Union[int, Tuple[int, int]],
                 feature_minimap: Union[int, Tuple[int, int]],
                 action_space: str = 'RAW',
                 feature_camera_width: int = 24,
                 use_feature_units: bool = True,
                 use_raw_units: bool = True,
                 use_camera_position: bool = True,
                 verbosity: int = 1,
                 parallel: int = -1,
                 dark: bool = True,
                 img_format: str = 'png',
                 generate_animation: bool = True,
                 animation_format: str = 'gif'):
        """
        Creates a new the location visualizer.
        :param int or Tuple[int, int] feature_screen: resolution for screen feature layers.
        :param int or Tuple[int, int]feature_minimap: resolution for minimap feature layers.
        :param int feature_camera_width: width of the feature layer camera.
        :param str action_space: action space for agent interface format.
        :param bool use_feature_units: whether to include feature_unit observations.
        :param bool use_raw_units: whether to include raw unit data in observations.
        :param bool use_camera_position: whether to include the camera's position (in minimap coordinates).
        :param int verbosity: level of logging verbosity.
        :param int parallel: how many replay files to process in parallel.
        :param bool dark: whether to use a dark theme/background for plotted figures.
        :param str img_format: image format for plotted figures.
        :param bool generate_animation: whether to generate an animation visualization.
        :param str animation_format: file format for animations (as in compatible with ffmpeg).
        """
        self._feature_screen_size = (feature_screen, feature_screen) \
            if isinstance(feature_screen, int) else feature_screen
        self._verbosity = verbosity
        self._parallel = parallel
        self._dark = dark
        self._img_format = img_format
        self._generate_animation = generate_animation
        self._animation_format = animation_format

        self._aif = features.parse_agent_interface_format(
            camera_width_world_units=feature_camera_width,
            use_camera_position=use_camera_position,
            use_feature_units=use_feature_units,
            use_raw_units=use_raw_units,
            action_space=action_space,
            feature_screen=feature_screen,
            feature_minimap=feature_minimap
        )

    def visualize_replays(self,
                          replays: str,
                          friendly_groups: Dict[str, np.ndarray],
                          enemy_groups: Dict[str, np.ndarray],
                          output_dir: str,
                          replay_sc2_version: Optional[str] = None):
        """
        Creates plots for the unit groups locations for the given set of replays.
        :param str replays: path to the replay file or replays directory from which to extract the units locations.
        :param dict[str, np.ndarray] friendly_groups: the groups of friendly unit types for which to track the location over episodes.
        :param dict[str, np.ndarray] enemy_groups: the groups of enemy unit types for which to track the location over episodes.
        :param str output_dir: path to the directory in which to save the results.
        :param str replay_sc2_version: SC2 version to use for replay.
        """
        # check histogram file
        histogram_file = os.path.join(output_dir, 'histogram_data.pkl.gz')
        if os.path.isfile(histogram_file):
            logging.info(f'Found histogram data file in {histogram_file}, loading...')
            histogram_data = load_object(histogram_file)
            logging.info(f'Loaded data for a total of {len(histogram_data[SELF][ALL_GROUP])} steps.')
        else:
            # check location data file
            location_file = os.path.join(output_dir, 'location_data.pkl.gz')
            if os.path.isfile(location_file):
                logging.info(f'Found location data file in {location_file}, loading...')
                location_data = load_object(location_file)
                logging.info(f'Loaded data for a total of {len(location_data)} episodes.')
            else:
                # gather all location data for the replays and save to file
                location_data = self._collect_location_data(replays, replay_sc2_version, friendly_groups, enemy_groups)
                logging.info(f'Saving location data to {location_file}...')
                save_object(location_data, location_file)

            # generate histogram data and save to file
            histogram_data = self._generate_histogram_data(location_data, friendly_groups, enemy_groups)
            logging.info(f'Saving histogram data to {histogram_file}...')
            save_object(histogram_data, histogram_file)

        # creates location plots for the different friendly vs enemy combinations

        # args = sorted(it.product([histogram_data], friendly_groups.keys(), enemy_groups.keys(), [output_dir]))
        args = sorted(it.product([histogram_data], [ALL_GROUP], [ALL_GROUP], [output_dir]))
        run_parallel(self._plot_comb_group_locations, args, processes=self._parallel, use_tqdm=True)

    def _collect_location_data(self, replays: str, replay_sc2_version: str,
                               friendly_groups: Dict[str, np.ndarray], enemy_groups: Dict[str, np.ndarray]) \
            -> List[SideGroupsHistogramList]:
        # creates and runs the replay processor
        results_queue = mp.JoinableQueue()
        extractor = LocationTrackingProcessor(friendly_groups, enemy_groups, results_queue, self._aif)
        runner = ReplayProcessRunner(replays, extractor, replay_sc2_version, self._parallel, player_ids=1)
        runner.run()

        # gather data from all replays
        location_data = []
        while True:
            try:
                data = results_queue.get(True, TIMEOUT)
            except queue.Empty:
                break
            if data is None:
                break
            logging.info(f'Got location data for {len(data[SELF][ALL_GROUP])} steps')
            location_data.append(data)

        logging.info('Done reading queue.')
        return location_data

    def _generate_histogram_data(self, location_data: List[Dict[PlayerRelative, Dict[str, List]]],
                                 friendly_groups: Dict[str, np.ndarray], enemy_groups: Dict[str, np.ndarray]) \
            -> SideGroupsHistogramList:
        # organizes location data by friendly and enemy groups
        logging.info('Organizing data by groups...')
        groups_data: SideGroupsHistogramList = {
            SELF: {g: [] for g in friendly_groups.keys()},
            ENEMY: {g: [] for g in enemy_groups.keys()},
            NEUTRAL: {ALL_GROUP: []}}
        for ep_data in location_data:
            for side, group_locs in ep_data.items():
                for g_name, g_locs in group_locs.items():
                    if len(g_locs) > 0:
                        groups_data[side][g_name].append(g_locs)

        # gets histograms for each unit group for each game step
        histogram_data: Dict[PlayerRelative, Dict[str, Any]] = {
            SELF: {g: [] for g in friendly_groups.keys()},
            ENEMY: {g: [] for g in enemy_groups.keys()},
            NEUTRAL: {ALL_GROUP: []}}
        args = []
        idx = 0
        for side, group_data in groups_data.items():
            for g_name, eps_loc_history in group_data.items():
                histogram_data[side][g_name] = idx  # remember index associated
                args.append(eps_loc_history)
                idx += 1

        logging.info(f'Computing histograms for each group and all steps ({len(args)} total)...')
        histograms = run_parallel(self._get_histograms, args, self._parallel, use_tqdm=True)
        for side, idxs in histogram_data.items():
            for g_name, idx in idxs.items():
                histogram_data[side][g_name] = histograms[idx]

        return histogram_data

    def _get_histograms(self, eps_loc_history: List) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
        # computes minimum step for which location data is available for this group
        locs_ts = [t_ranges[0][0] for ep_loc_hist in eps_loc_history for loc, t_ranges in ep_loc_hist.items()]
        if len(locs_ts) == 0:
            return None
        min_t = min(locs_ts)

        # gets histograms and alpha maps for each step
        return [self._get_histogram(eps_loc_history, max(t, min_t)) for t in np.linspace(0, 1, NUM_FRAMES)]

    def _get_histogram(self, eps_loc_history: List, t_max: float) -> Tuple[np.ndarray, np.ndarray]:
        # initialize histograms
        histogram = np.zeros(np.array(self._feature_screen_size) + 1)
        counts = np.zeros(np.array(self._feature_screen_size) + 1)

        if len(eps_loc_history) == 0:
            return histogram, counts  # no data to process

        # registers where and when units have been up until max_t in each episode
        min_dt = 1. / NUM_FRAMES
        for ep_loc_history in eps_loc_history:
            for loc, t_ranges in ep_loc_history.items():
                loc_ts = [min(t_max, t_range[1]) for t_range in t_ranges if t_max >= t_range[0]]
                if len(loc_ts) == 0:
                    continue
                loc_t = max(loc_ts)
                loc_count = sum([min(t_max, t_range[1]) - t_range[0] + min_dt
                                 for t_range in t_ranges if t_max >= t_range[0]])
                np.add.at(histogram, loc, loc_t * loc_count)
                np.add.at(counts, loc, loc_count)

        return histogram, counts

    def _plot_comb_group_locations(self, groups_histograms: Dict[PlayerRelative, Dict[str, List]],
                                   friendly_group: str, enemy_group: str, output_dir: str):
        title = f'Friendly {friendly_group.title()} vs Enemy {enemy_group.title()} '
        logging.info(f'Processing frames for {title}...')
        groups_histograms = {SELF: groups_histograms[SELF][friendly_group],
                             ENEMY: groups_histograms[ENEMY][enemy_group],
                             NEUTRAL: groups_histograms[NEUTRAL][ALL_GROUP]}  # always present neutral

        # saves both single image and animation
        file_name = os.path.join(output_dir, f'{friendly_group.lower()}-vs-{enemy_group.lower()}')
        self._save_image(groups_histograms, title, f'{file_name}.{self._img_format}')
        if self._generate_animation:
            self._save_animation(groups_histograms, title, f'{file_name}.{self._animation_format}')
        gc.collect()

    def _save_animation(self, groups_histograms: Dict[PlayerRelative, List], title: str, output_video: str):
        if groups_histograms[SELF] is None and groups_histograms[ENEMY] is None and groups_histograms[NEUTRAL] is None:
            return
        if os.path.exists(output_video):
            os.remove(output_video)  # check existing file
            time.sleep(0.5)

        # as suggested in https://superuser.com/a/556031
        with skvideo.io.FFmpegWriter(output_video, inputdict={'-r': str(ANIM_FPS)},
                                     outputdict={'-vf': 'split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse'}) as writer:
            for t in tqdm.tqdm(range(NUM_FRAMES)):
                # renders and saves frame
                frame = self._render_image(groups_histograms, t, None, title)
                writer.writeFrame(frame)
                writer._proc.stdin.flush()
                del frame

    def _save_image(self, groups_histograms, title, output_img=None):
        if groups_histograms[SELF] is None and groups_histograms[ENEMY] is None and groups_histograms[NEUTRAL] is None:
            return
        self._render_image(groups_histograms, -1, output_img, title)

    def _render_image(self, groups_histograms: Dict[PlayerRelative, List], t_index: int,
                      output_img: Optional[str], title: str):

        def _get_histogram(side: PlayerRelative) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            if groups_histograms[side] is not None:
                histogram, counts = groups_histograms[side][t_index]
                # friendly_counts = groups_histograms[SELF][t_index]
                # enemy_counts = groups_histograms[SELF][t_index]
                with np.errstate(divide='ignore', invalid='ignore'):
                    histogram = np.nan_to_num(histogram / counts)  # normalize histogram
                    counts_norm = counts / np.max(counts)
                    # alpha = 1 - np.exp(-ALPHA_EXP_FACTOR * counts_norm)  # compute alpha
                    # hist, buckets = np.histogram(counts, 1000)
                    # cdf = np.cumsum(hist / np.sum(hist))
                    # thresh = buckets[np.where(cdf > 0.8)[0][0]]
                    # alpha = np.clip(np.nan_to_num(counts / thresh), 0, 1)
                    alpha = np.tanh(100 * counts_norm)
                return histogram, alpha
                # return counts_norm , alpha
            return None, None  # no data

        def _plot_histogram(histogram: np.ndarray, alpha: np.ndarray, color_map, zorder: int):
            if histogram is not None:
                plt.pcolormesh(histogram.T, cmap=color_map, alpha=alpha.T, zorder=zorder, rasterized=True)

        # gather data across episodes for all timesteps until t_index
        friendly_histogram, friendly_alpha = _get_histogram(SELF)
        enemy_histogram, enemy_alpha = _get_histogram(ENEMY)
        neutral_histogram, neutral_alpha = _get_histogram(NEUTRAL)

        # friendly on top, so make semi-transparent where intersects
        if friendly_histogram is not None and (enemy_histogram is not None or neutral_histogram is not None):
            enemy_or = np.logical_or(enemy_histogram > 0, neutral_histogram > 0) \
                if enemy_histogram is not None and neutral_histogram is not None else \
                enemy_histogram > 0 if enemy_histogram is not None else neutral_histogram > 0
            intersect = np.where(np.logical_and(friendly_histogram > 0, enemy_or))
            other_alpha = enemy_alpha if enemy_histogram is not None else neutral_alpha
            # friendly_alpha[intersect] = ((1 - other_alpha[intersect]) + friendly_alpha[intersect]) * 0.5
            friendly_alpha[intersect] = np.maximum(0.3,
                                                   np.minimum(1 - other_alpha[intersect], friendly_alpha[intersect]))
            # friendly_alpha[intersect] = np.minimum(INTERSECTION_ALPHA, friendly_alpha[intersect])

        # plot locations by time using a different color for friendly / enemy / neutral units
        if self._dark:
            plt.style.use('dark_background')
        fig, ax = plt.subplots()
        _plot_histogram(friendly_histogram, friendly_alpha, FRIENDLY_COLOR_MAP, 3)
        _plot_histogram(enemy_histogram, enemy_alpha, ENEMY_COLOR_MAP, 2)
        _plot_histogram(neutral_histogram, neutral_alpha, NEUTRAL_CMAP_DARK if self._dark else NEUTRAL_CMAP, 1)
        gradient_line_legend([FRIENDLY_COLOR_MAP, ENEMY_COLOR_MAP], ['Friendly', 'Enemy'])

        # formats plot
        plt.title(title)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, which='both', linestyle='--', color='dimgrey' if self._dark else 'lightgrey')
        ax.xaxis.grid(True, which='both', linestyle='--', color='dimgrey' if self._dark else 'lightgrey')
        ax.invert_yaxis()  # y coordinates are inverted on map
        fig.tight_layout(pad=0)

        # save plt to file or get rgba image array
        img = None
        if output_img is None:
            with io.BytesIO() as io_buf:
                plt.savefig(io_buf, format='rgba')  # pad_inches=0, bbox_inches='tight', dpi=DPI)
                io_buf.seek(0)
                img = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
                img = np.reshape(img, newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        else:
            plt.savefig(output_img, pad_inches=0, bbox_inches='tight', dpi=DPI)

        # restore default theme and clear
        if self._dark:
            plt.style.use('default')
        fig.clear()
        plt.close(fig)
        return img
