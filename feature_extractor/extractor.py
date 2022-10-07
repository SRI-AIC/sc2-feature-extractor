import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from s2clientprotocol import sc2api_pb2 as sc_pb
from pysc2.env.environment import TimeStep
from pysc2.lib.actions import FunctionCall
from pysc2.lib.features import AgentInterfaceFormat
from feature_extractor.extractors import FRIENDLY_STR, ENEMY_STR, MetaExtractor, EPISODE_STR, TIME_STEP_STR, \
    FeatureExtractor
from feature_extractor.replayer import DebugReplayProcessor, DebugStepListener
from feature_extractor.util.io import get_file_name_without_extension

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class ExtractorListener(DebugStepListener):
    """
    A high-level feature extractor for StarCraft II using pysc2 framework that processes data from individual replays.
    """

    def __init__(self,
                 meta_extractor: MetaExtractor,
                 extractors: Dict[str, List[FeatureExtractor]],
                 output_dir: str):
        """
        Creates a new feature extractor.
        :param MetaExtractor meta_extractor: the metadata extractor, responsible for identifying the episode, step, etc.
        :param dict[str, list[FeatureExtractor]] extractors: the list of extractors, separated by side (friendly, enemy).
        :param str output_dir: the path to the directory in which to save the features' history file.
        """
        self.meta_extractor = meta_extractor
        self.feature_extractors = extractors
        self.output_dir = output_dir

        # creates list of feature labels from each extractor
        self.feature_labels = {}
        for side in [FRIENDLY_STR, ENEMY_STR]:
            self.feature_labels[side] = []
            for extractor in self.feature_extractors[side]:
                self.feature_labels[side].extend(extractor.features_labels())

        # current replay info
        self.replay_path: str = ''
        self.output_file: str = ''  # set by the processor
        self.feature_history: Dict[str, List[List[Union[bool, int, float, str]]]] = {}
        self.side: str = FRIENDLY_STR
        self.num_players: int = 0
        self._total_steps: int = 0
        self._ep: int = -1
        self._ep_step: int = 0

    def start_replay(self, replay_path: str, replay_info: sc_pb.ResponseReplayInfo, player_perspective: int):
        """
        Called when starting a new replay.
        :param str replay_path: replay file name.
        :param ResponseReplayInfo replay_info: protobuf message.
        :param int player_perspective: ID of player whose perspective we see observations.
        """
        self._ep_step = 0

        # checks whether the replay is different/new
        if self.replay_path != replay_path:
            self.replay_path = replay_path
            self.meta_extractor.replay_name = get_file_name_without_extension(replay_path)
            self.num_players = len(replay_info.player_info)
            self.feature_history = {FRIENDLY_STR: [], ENEMY_STR: []}
            self._ep = 0

        # checks side
        self.side = FRIENDLY_STR if player_perspective == self.meta_extractor.config.friendly_id else ENEMY_STR
        logging.info(f'Extracting features from \'{replay_path}\' in player {player_perspective}\'s '
                     f'perspective ({self.side} side)...')

    def reset(self, pb_obs: sc_pb.ResponseObservation, agent_obs: TimeStep):
        """
        Called for the first observation of the replay. Resets all feature extractors.
        :param ResponseObservation pb_obs: the observation in protobuf form.
        :param TimeStep agent_obs: the observation in pysc2 features form.
        :return:
        """
        for extractor in self.feature_extractors[self.side]:
            extractor.reset(agent_obs)

    def step(self,
             ep: int,
             step: int,
             pb_obs: sc_pb.ResponseObservation,
             agent_obs: TimeStep,
             agent_actions: List[FunctionCall]):
        """
        Updates the feature extractor with the given observations.
        :param int ep: the episode that this observation was made.
        :param int step: the episode time-step in which this observation was made.
        :param ResponseObservation pb_obs: the observation in protobuf form.
        :param TimeStep agent_obs: the observation in pysc2 features form.
        :param list[FunctionCall] agent_actions: list of actions executed by the agent between the previous observation
        and the current observation.
        :return:
        """
        self._ep = ep
        self._ep_step = step

        # ignore non-sampling steps
        if self._ep_step % self.meta_extractor.config.sample_int:
            return

        # update features from each extractor
        features = []
        for extractor in self.feature_extractors[self.side]:
            features.extend(extractor.extract(self._ep, step, agent_obs.observation, pb_obs))

        self.feature_history[self.side].append(features)
        self._total_steps += 1

    def finish_replay(self):
        """
        Saves the features history to a CSV file.
        """
        # checks data, wait for all replays to be completed and sides processed
        if len(self.feature_history[FRIENDLY_STR]) == 0 or \
                (self.num_players == 2 and len(self.feature_history[ENEMY_STR]) == 0):
            return

        if self.num_players == 2:
            # stacks features for both sides together
            data_set = np.hstack((np.array(self.feature_history[FRIENDLY_STR]),
                                  np.array(self.feature_history[ENEMY_STR])))

            # gets names for each column
            header = []
            for side in [FRIENDLY_STR, ENEMY_STR]:
                header.extend(self.feature_labels[side])
        else:
            data_set = self.feature_history[FRIENDLY_STR]
            header = self.feature_labels[FRIENDLY_STR]

        df = pd.DataFrame(data_set, columns=header)
        df[EPISODE_STR] = df[EPISODE_STR].astype(int)
        df[TIME_STEP_STR] = df[TIME_STEP_STR].astype(int)
        df.sort_values([EPISODE_STR, TIME_STEP_STR], inplace=True, ascending=[True, True])  # sanity check
        df.to_csv(self.output_file, index=False)
        logging.info(f'Finished on episode {self._ep} ({int(self._total_steps / 2)} total steps), '
                     f'saved results to {self.output_file}.')


class ExtractorProcessor(DebugReplayProcessor):
    """
    Tells the processing code how to configure the game, and instantiates the listeners that actually process the data.
    """

    def __init__(self,
                 extractor_listener: ExtractorListener,
                 agent_interface_format: AgentInterfaceFormat,
                 ignore_existing: bool = True):
        """
        Creates a new processor.
        :param DebugStepListener extractor_listener: the listener used to step through the replay and extract the features.
        :param AgentInterfaceFormat agent_interface_format: the agent's pysc2 interface format.
        :param bool ignore_existing: whether to ignore the replay if the output CSV file already exists. Useful for
        resuming feature extraction.
        """
        self._agent_interface_format = agent_interface_format
        self.extractor = extractor_listener
        self.ignore_existing = ignore_existing

    @property
    def interface(self) -> sc_pb.InterfaceOptions:
        aif = self._agent_interface_format
        interface = sc_pb.InterfaceOptions(
            raw=True, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=aif.camera_width_world_units))
        if aif.rgb_dimensions is not None:
            aif.rgb_dimensions.screen.assign_to(interface.render.resolution)
            aif.rgb_dimensions.minimap.assign_to(interface.render.minimap_resolution)
        aif.feature_dimensions.screen.assign_to(interface.feature_layer.resolution)
        aif.feature_dimensions.minimap.assign_to(interface.feature_layer.minimap_resolution)
        return interface

    @property
    def agent_interface_format(self) -> Optional[AgentInterfaceFormat]:
        return self._agent_interface_format

    def create_listeners(self) -> List[DebugStepListener]:
        """
        Returns a list of Listeners that will process the actual replay data. This has to be deferred because the
        replays are processed in sub-processes. This way we can have different Listener instances for each process and
        we don't have to worry about concurrency.
        """
        return [self.extractor]

    def valid_replay(self, info, ping, replay_path) -> bool:
        self.extractor.output_file = os.path.join(
            self.extractor.output_dir, f'{get_file_name_without_extension(replay_path)}.csv')

        # no need to extract since file already exists
        return not self.ignore_existing or not os.path.isfile(self.extractor.output_file)
