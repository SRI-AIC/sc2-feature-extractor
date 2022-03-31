import numpy as np
from typing import List, Union
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from pysc2.lib.named_array import NamedDict
from pysc2.lib.features import PlayerRelative
from feature_extractor.config import FeatureExtractorConfig
from feature_extractor.extractors import FeatureExtractor, FeatureType, FeatureDescriptor, FRIENDLY_STR, ENEMY_STR

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class UnitGroupExtractor(FeatureExtractor):
    """
    An extractor that detects the presence of friendly and enemy unit groups (boolean features).
    """

    def __init__(self, config: FeatureExtractorConfig, categorical: bool):
        super().__init__(config)
        self._categorical = categorical
        self._friendly_filter = self._convert_unit_filter(self.config.unit_group_friendly_filter)
        self._enemy_filter = self._convert_unit_filter(self.config.unit_group_enemy_filter)

    def features_labels(self) -> List[str]:
        labels = []
        for g_name in self._friendly_filter:
            labels.append(f'Present_{FRIENDLY_STR}_{g_name}')
        for g_name in self._enemy_filter:
            labels.append(f'Present_{ENEMY_STR}_{g_name}')
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        if self._categorical:
            return [FeatureDescriptor(lbl, FeatureType.Boolean) for lbl in self.features_labels()]
        else:
            descriptors = []
            for g_name, group in self._friendly_filter.items():
                num_units = self.config.max_friendly_units
                max_num = sum(num_units[u] if u in num_units else 0 for u in group)
                descriptors.append(FeatureDescriptor(f'Present_{FRIENDLY_STR}_{g_name}',
                                                     FeatureType.Integer, [0, max_num]))
            for g_name, group in self._enemy_filter.items():
                num_units = self.config.max_enemy_units
                max_num = sum(num_units[u] if u in num_units else 0 for u in group)
                descriptors.append(FeatureDescriptor(f'Present_{ENEMY_STR}_{g_name}',
                                                     FeatureType.Integer, [0, max_num]))
            return descriptors

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        def _update_features(filter, unit_types):
            if self._categorical:
                unit_types = list(set(unit_types))  # only interested in unit types, not amount
            for group in filter.values():
                if self._categorical:
                    # at least one unit of the group should be on the environment
                    features.append(np.any(np.in1d(group, unit_types, assume_unique=True)))
                else:
                    # count all units of this group
                    features.append(np.sum(np.in1d(unit_types, group, assume_unique=False)))

        # get units for each faction
        alliance = obs['raw_units'][:, 'alliance']

        # update features
        features = []
        unit_types = obs['raw_units'][np.where(alliance == PlayerRelative.SELF)][:, 'unit_type']
        _update_features(self._friendly_filter, unit_types)

        unit_types = obs['raw_units'][np.where(alliance != PlayerRelative.SELF)][:, 'unit_type']
        _update_features(self._enemy_filter, unit_types)

        return features
