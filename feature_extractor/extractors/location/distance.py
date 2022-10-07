import numpy as np
import itertools as it
from typing import List, Union
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from pysc2.lib.named_array import NamedDict
from pysc2.lib.features import PlayerRelative
from feature_extractor.config import FeatureExtractorConfig
from feature_extractor.extractors import FeatureExtractor, DEFAULT_FEATURE_VAL, FeatureType, FeatureDescriptor
from feature_extractor.extractors.location import get_unit_locations

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

MELEE_STR = 'melee'
CLOSE_STR = 'close'
FAR_STR = 'far'


class DistanceExtractor(FeatureExtractor):
    """
    An extractor that detects the distance between friendly and enemy unit groups, measured as the minimal distance
    between any two units of each force in those groups.
    """

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self._friendly_filter = self._convert_unit_filter(self.config.distance_friendly_filter)
        self._enemy_filter = self._convert_unit_filter(self.config.distance_enemy_filter)
        self._group_combs = list(it.product(self._friendly_filter.keys(), self._enemy_filter.keys()))

    def features_labels(self) -> List[str]:
        labels = []
        if self.config.distance_categorical:
            labels = [f'DistanceCat_{fg}_{eg}' for fg, eg in self._group_combs]
        if self.config.distance_numeric:
            labels.extend([f'Distance_{fg}_{eg}' for fg, eg in self._group_combs])
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        descriptors = []
        if self.config.distance_categorical:
            labels = [f'DistanceCat_{fg}_{eg}' for fg, eg in self._group_combs]
            feat_values = [DEFAULT_FEATURE_VAL, MELEE_STR, CLOSE_STR, FAR_STR]
            descriptors = [FeatureDescriptor(lbl, FeatureType.Categorical, feat_values) for lbl in labels]
        if self.config.distance_numeric:
            labels = [f'Distance_{fg}_{eg}' for fg, eg in self._group_combs]
            descriptors.extend([FeatureDescriptor(lbl, FeatureType.Real, [0., 1.]) for lbl in labels])
        return descriptors

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        # gets locations of units for each faction and group combination
        friendly_locs = get_unit_locations(obs, self._friendly_filter, PlayerRelative.SELF)
        enemy_locs = get_unit_locations(obs, self._enemy_filter, PlayerRelative.SELF, True)

        # get minimal distance between any 2 units of each faction for all group combinations
        min_dists = []
        for fg, eg in self._group_combs:
            min_dist = np.finfo(np.float).max
            for friendly_loc, enemy_loc in it.product(friendly_locs[fg], enemy_locs[eg]):
                min_dist = min(min_dist, np.linalg.norm(friendly_loc - enemy_loc))
            min_dists.append(min_dist)

        # update distance to enemy feature according to thresholds
        dists = []
        map_size = pb_obs.observation.raw_data.map_state.visibility.size
        max_len = np.linalg.norm([map_size.x, map_size.y])

        if self.config.distance_categorical:
            for min_dist in min_dists:
                dist = DEFAULT_FEATURE_VAL
                if min_dist <= max_len * self.config.melee_range_ratio:
                    dist = MELEE_STR
                elif min_dist <= max_len * self.config.close_range_ratio:
                    dist = CLOSE_STR
                elif min_dist <= max_len * self.config.far_range_ratio:
                    dist = FAR_STR
                dists.append(dist)

        if self.config.distance_numeric:
            for min_dist in min_dists:
                dists.append(np.nan if min_dist == np.finfo(np.float).max else min(1, min_dist / max_len))

        return dists
