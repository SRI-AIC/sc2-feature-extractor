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

LOW_STR = 'low'
MEDIUM_STR = 'medium'
HIGH_STR = 'high'


class ElevationExtractor(FeatureExtractor):
    """
    An extractor that computes the mean elevation of friendly and enemy groups of units.
    """

    def __init__(self, config: FeatureExtractorConfig, categorical: bool):
        super().__init__(config)
        self._categorical = categorical
        self._friendly_filter = self._convert_unit_filter(self.config.elevation_friendly_filter)
        self._enemy_filter = self._convert_unit_filter(self.config.elevation_enemy_filter)

    def features_labels(self) -> List[str]:
        labels = []
        for g_name in self._friendly_filter:
            labels.append(f'Elevation_{FRIENDLY_STR}_{g_name}')
        for g_name in self._enemy_filter:
            labels.append(f'Elevation_{ENEMY_STR}_{g_name}')
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        if self._categorical:
            feat_values = [DEFAULT_FEATURE_VAL, LOW_STR, MEDIUM_STR, HIGH_STR]
            return [FeatureDescriptor(lbl, FeatureType.Categorical, feat_values) for lbl in self.features_labels()]
        else:
            return [FeatureDescriptor(lbl, FeatureType.Real, [0, np.finfo(np.float).max])
                    for lbl in self.features_labels()]

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        def get_avg_height(units_locs):
            return np.nan if len(units_locs) == 0 else \
                np.mean([height_map[tuple(loc)] for loc in units_locs])

        features = []
        height_map = np.array(obs['feature_screen']['height_map']).T

        def _add_groups_features(_filter):
            # updates elevation feature according to avg elevation of units in each group
            for g in _filter:
                height = get_avg_height(locs[g])
                if not self._categorical:
                    features.append(height)
                elif height <= self.config.low_elevation:
                    features.append(LOW_STR)
                elif height <= self.config.medium_elevation:
                    features.append(MEDIUM_STR)
                elif height <= self.config.high_elevation:
                    features.append(HIGH_STR)
                else:
                    features.append(DEFAULT_FEATURE_VAL)

        # gets locations of units for each group of each faction
        locs = get_unit_locations(obs, self._friendly_filter, PlayerRelative.SELF)
        _add_groups_features(self._friendly_filter)

        locs = get_unit_locations(obs, self._enemy_filter, PlayerRelative.SELF, True)
        _add_groups_features(self._enemy_filter)

        return features
