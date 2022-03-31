import numpy as np
from typing import List, Union
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from pysc2.lib.named_array import NamedDict
from pysc2.lib.features import PlayerRelative
from feature_extractor.config import FeatureExtractorConfig
from feature_extractor.extractors import FeatureExtractor, FRIENDLY_STR, ENEMY_STR, DEFAULT_FEATURE_VAL, FeatureType, \
    FeatureDescriptor
from feature_extractor.extractors.location import get_unit_locations

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

COMPACT_STR = 'compact'
SPREAD_STR = 'spread'
SCATTERED_STR = 'scattered'


class ConcentrationExtractor(FeatureExtractor):
    """
    An extractor that computes how concentrated/compact the friendly and enemy forces are, calculated according to the
    average pairwise distance of units within each force.
    """

    def __init__(self, config: FeatureExtractorConfig, categorical: bool):
        super().__init__(config)
        self._categorical = categorical
        self._friendly_filter = self._convert_unit_filter(self.config.concentration_friendly_filter)
        self._enemy_filter = self._convert_unit_filter(self.config.concentration_enemy_filter)

    def features_labels(self) -> List[str]:
        labels = []
        for g_name in self._friendly_filter:
            labels.append(f'Concentration_{FRIENDLY_STR}_{g_name}')
        for g_name in self._enemy_filter:
            labels.append(f'Concentration_{ENEMY_STR}_{g_name}')
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        if self._categorical:
            feat_values = [DEFAULT_FEATURE_VAL, COMPACT_STR, SPREAD_STR, SCATTERED_STR]
            return [FeatureDescriptor(lbl, FeatureType.Categorical, feat_values)
                    for lbl in self.features_labels()]
        else:
            return [FeatureDescriptor(lbl, FeatureType.Real, [0., 1.]) for lbl in self.features_labels()]

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        def get_avg_within_dist(units_locs):
            if len(units_locs) == 0:
                return np.nan
            if len(units_locs) == 1:
                return 0.

            dists = []
            for i in range(len(units_locs)):
                for j in range(i + 1, len(units_locs)):
                    dists.append(np.linalg.norm(units_locs[i] - units_locs[j]))
            return np.mean(dists)

        features = []
        map_size = pb_obs.observation.raw_data.map_state.visibility.size
        max_len = np.linalg.norm([map_size.x, map_size.y])

        def _add_groups_features(_filter):
            # update concentration feature according to thresholds to avg distance between 2 units within each group
            for g in _filter:
                avg_dist = get_avg_within_dist(locs[g])
                if not self._categorical:
                    features.append(min(1, avg_dist / max_len))  # gets the avg distance ratio
                elif avg_dist <= max_len * self.config.compact_ratio:
                    features.append(COMPACT_STR)
                elif avg_dist <= max_len * self.config.spread_ratio:
                    features.append(SPREAD_STR)
                elif avg_dist <= max_len * self.config.scattered_ratio:
                    features.append(SCATTERED_STR)
                else:
                    features.append(DEFAULT_FEATURE_VAL)

        # gets locations of units for each group of each faction
        locs = get_unit_locations(obs, self._friendly_filter, PlayerRelative.SELF)
        _add_groups_features(self._friendly_filter)

        locs = get_unit_locations(obs, self._enemy_filter, PlayerRelative.SELF, True)
        _add_groups_features(self._enemy_filter)

        return features
