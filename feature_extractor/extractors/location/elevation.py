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

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self._friendly_filter = self._convert_unit_filter(self.config.elevation_friendly_filter)
        self._enemy_filter = self._convert_unit_filter(self.config.elevation_enemy_filter)

    def features_labels(self) -> List[str]:
        labels = []
        if self.config.elevation_categorical:
            labels.extend([f'ElevationCat_{FRIENDLY_STR}_{g_name}' for g_name in self._friendly_filter])
            labels.extend([f'ElevationCat_{ENEMY_STR}_{g_name}' for g_name in self._enemy_filter])
        if self.config.elevation_numeric:
            labels.extend([f'Elevation_{FRIENDLY_STR}_{g_name}' for g_name in self._friendly_filter])
            labels.extend([f'Elevation_{ENEMY_STR}_{g_name}' for g_name in self._enemy_filter])
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        descriptors = []
        if self.config.elevation_categorical:
            feat_values = [DEFAULT_FEATURE_VAL, LOW_STR, MEDIUM_STR, HIGH_STR]
            descriptors.extend([FeatureDescriptor(
                f'ElevationCat_{FRIENDLY_STR}_{g_name}', FeatureType.Categorical, feat_values)
                for g_name in self._friendly_filter])
            descriptors.extend([FeatureDescriptor(
                f'ElevationCat_{ENEMY_STR}_{g_name}', FeatureType.Categorical, feat_values)
                for g_name in self._enemy_filter])
        if self.config.elevation_numeric:
            descriptors.extend([FeatureDescriptor(
                f'Elevation_{FRIENDLY_STR}_{g_name}', FeatureType.Real, [0, np.finfo(np.float).max])
                for g_name in self._friendly_filter])
            descriptors.extend([FeatureDescriptor(
                f'Elevation_{ENEMY_STR}_{g_name}', FeatureType.Real, [0, np.finfo(np.float).max])
                for g_name in self._enemy_filter])
        return descriptors

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        def _get_avg_height(units_locs):
            return np.nan if len(units_locs) == 0 else \
                np.mean([height_map[tuple(loc)] for loc in units_locs])

        def _add_groups_features(_filter, locs, cat):
            # updates elevation feature according to avg elevation of units in each group
            for g in _filter:
                height = _get_avg_height(locs[g])
                if not cat:
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
        friendly_locs = get_unit_locations(obs, self._friendly_filter, PlayerRelative.SELF)
        enemy_locs = get_unit_locations(obs, self._enemy_filter, PlayerRelative.SELF, True)
        height_map = np.array(obs['feature_screen']['height_map']).T

        features = []
        if self.config.elevation_categorical:
            _add_groups_features(self._friendly_filter, friendly_locs, cat=True)
            _add_groups_features(self._enemy_filter, enemy_locs, cat=True)
        if self.config.elevation_numeric:
            _add_groups_features(self._friendly_filter, friendly_locs, cat=False)
            _add_groups_features(self._enemy_filter, enemy_locs, cat=False)

        return features
