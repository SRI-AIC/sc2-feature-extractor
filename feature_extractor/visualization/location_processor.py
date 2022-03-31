import logging
import multiprocessing as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
from pysc2.env.environment import TimeStep
from pysc2.lib.features import PlayerRelative
from pysc2.env.sc2_env import AgentInterfaceFormat
from s2clientprotocol import sc2api_pb2 as sc_pb
from feature_extractor.replayer import DebugReplayProcessor, DebugStepListener
from feature_extractor.extractors.location import get_unit_locations

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

ALL_GROUP = 'All'


class LocationTrackingProcessor(DebugReplayProcessor):
    """
    Tells the processing code how to configure the game, and instantiates the listeners that actually process the data.
    """

    def __init__(self, friendly_groups: Dict[str, np.ndarray], enemy_groups: Dict[str, np.ndarray],
                 results_queue: mp.Queue, agent_interface_format: AgentInterfaceFormat, friendly_id=1):
        """
        Creates a new processor.
        :param Dict[str, np.ndarray] friendly_groups: the groups of friendly unit types for which to track the location over episodes.
        :param Dict[str, np.ndarray] enemy_groups: the groups of enemy unit types for which to track the location over episodes.
        :param mp.Queue results_queue: the queue to put data in.
        :param AgentInterfaceFormat agent_interface_format: the agent's pysc2 interface format.
        :param int friendly_id: the id of the player that we consider to be the "friendly" faction.
        """
        self._aif = agent_interface_format

        self._interface_options = sc_pb.InterfaceOptions(
            raw=True, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=self._aif.camera_width_world_units))
        self._aif.feature_dimensions.screen.assign_to(self._interface_options.feature_layer.resolution)
        self._aif.feature_dimensions.minimap.assign_to(self._interface_options.feature_layer.minimap_resolution)
        if self._aif.rgb_dimensions is not None:
            self._aif.rgb_dimensions.screen.assign_to(self._interface_options.render.resolution)
            self._aif.rgb_dimensions.minimap.assign_to(self._interface_options.render.minimap_resolution)

        self._listener = LocationTrackerListener(friendly_groups, enemy_groups, results_queue, friendly_id)

    @property
    def interface(self) -> sc_pb.InterfaceOptions:
        return self._interface_options

    @property
    def agent_interface_format(self) -> Optional[AgentInterfaceFormat]:
        return self._aif

    def create_listeners(self) -> List[DebugStepListener]:
        return [self._listener]


class LocationTrackerListener(DebugStepListener):
    """
    A location tracker for unit types during SC2 episode replays.
    """
    _replay_name: str
    _friendly_locs: List[Tuple[str, PlayerRelative, float, Dict[str, np.ndarray]]]
    _enemy_locs: List[Tuple[str, PlayerRelative, float, Dict[str, np.ndarray]]]
    _neutral_locs: List[Tuple[str, PlayerRelative, float, Dict[str, np.ndarray]]]

    def __init__(self, friendly_groups: Dict[str, np.ndarray], enemy_groups: Dict[str, np.ndarray],
                 results_queue: mp.Queue, friendly_id=1):
        """
        Creates a new location tracker listener.
        :param Dict[str, np.ndarray] friendly_groups: the groups of friendly unit types for which to track the location over episodes.
        :param Dict[str, np.ndarray] enemy_groups: the groups of enemy unit types for which to track the location over episodes.
        :param mp.Queue results_queue: the queue to put data in.
        :param int friendly_id: the id of the player that we consider to be the "friendly" faction.
        """
        self._friendly_groups = friendly_groups
        self._enemy_groups = enemy_groups
        self._results_queue = results_queue
        self._friendly_id = friendly_id

    def start_replay(self, replay_name, replay_info, player_perspective):
        self._replay_name = replay_name
        self._friendly_locs = []
        self._enemy_locs = []
        self._neutral_locs = []
        logging.info(f'[LocationTracker] Collecting data from replay \'{replay_name}\'...')

    def finish_replay(self):
        self._send_locations()
        logging.info(f'[LocationTracker] Finished processing replay \'{self._replay_name}\'')

    def reset(self, pb_obs, agent_obs):
        pass

    def step(self, ep, step, pb_obs, agent_obs, agent_actions):
        """
        Tracks location of groups of units.
        :param int ep: the episode that this observation was made.
        :param int step: the episode time-step in which this observation was made.
        :param ResponseObservation pb_obs: the observation in protobuf form.
        :param TimeStep agent_obs: the observation in pysc2 features form.
        :param list[FunctionCall] agent_actions: list of actions executed by the agent between the previous observation
        and the current observation.
        :return:
        """
        # checks new episode, put data into queue
        if step == 0 and len(self._friendly_locs) > 0:
            self._send_locations()

        # record location for each group of units
        friendly_locs = get_unit_locations(
            agent_obs.observation, self._friendly_groups, PlayerRelative.SELF, raw_units=False)
        self._friendly_locs.append((self._replay_name, PlayerRelative.SELF, step, friendly_locs))
        enemy_locs = get_unit_locations(
            agent_obs.observation, self._enemy_groups, PlayerRelative.ENEMY, raw_units=False)
        self._enemy_locs.append((self._replay_name, PlayerRelative.ENEMY, step, enemy_locs))
        neutral_locs = get_unit_locations(
            agent_obs.observation, {ALL_GROUP: self._enemy_groups[ALL_GROUP]}, PlayerRelative.NEUTRAL, raw_units=False)
        self._neutral_locs.append((self._replay_name, PlayerRelative.NEUTRAL, step, neutral_locs))

    def _send_locations(self):
        # normalize steps
        friendly_locations = self._filter_locations(self._friendly_locs, self._friendly_groups)
        enemy_locations = self._filter_locations(self._enemy_locs, self._enemy_groups)
        neutral_locations = self._filter_locations(self._neutral_locs, self._enemy_groups)
        self._results_queue.put({PlayerRelative.SELF: friendly_locations,
                                 PlayerRelative.ENEMY: enemy_locations,
                                 PlayerRelative.NEUTRAL: neutral_locations})
        self._friendly_locs = []
        self._enemy_locs = []

    @staticmethod
    def _filter_locations(locations, unit_filter):

        def _normalize_step(step):
            return (step + 1) / len(locations)  # avoid step 0.0

        # creates structure that tracks units locations and creates dictionary
        # {group_x: { pos_xy : [[start0, end0], [start1, end1], ...],
        #   ... } }
        new_locations = {g_name: {} for g_name in unit_filter.keys()}
        for name, side, step, locs in locations:
            for g_name, g_locs in locs.items():
                if len(g_locs) == 0:
                    continue  # ignore if no group locs
                for g_loc in g_locs:
                    g_loc = tuple(g_loc)
                    if g_loc not in new_locations[g_name]:
                        new_locations[g_name][g_loc] = [[step, step]]  # new loc, new sequence
                    else:
                        last_step = new_locations[g_name][g_loc][-1][1]
                        if last_step == step - 1:
                            new_locations[g_name][g_loc][-1][1] = step  # continue sequence
                        else:
                            new_locations[g_name][g_loc].append([step, step])  # new sequence

        # normalize steps
        for g_name, g_data in new_locations.items():
            for g_loc, g_ranges in g_data.items():
                new_locations[g_name][g_loc] = [[_normalize_step(r[0]), _normalize_step(r[1])] for r in g_ranges]
        return new_locations
