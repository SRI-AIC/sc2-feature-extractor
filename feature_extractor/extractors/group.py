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

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self._friendly_filter = self._convert_unit_filter(self.config.unit_group_friendly_filter)
        self._enemy_filter = self._convert_unit_filter(self.config.unit_group_enemy_filter)

    def features_labels(self) -> List[str]:
        labels = []
        if self.config.unit_group_categorical:
            labels.extend([f'Present_{FRIENDLY_STR}_{g_name}' for g_name in self._friendly_filter])
            labels.extend([f'Present_{ENEMY_STR}_{g_name}' for g_name in self._enemy_filter])
        if self.config.unit_group_numeric:
            labels.extend([f'Number_{FRIENDLY_STR}_{g_name}' for g_name in self._friendly_filter])
            labels.extend([f'Number_{ENEMY_STR}_{g_name}' for g_name in self._enemy_filter])
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        descriptors = []
        if self.config.unit_group_categorical:
            descriptors.extend([FeatureDescriptor(f'Present_{FRIENDLY_STR}_{g_name}', FeatureType.Boolean)
                                for g_name in self._friendly_filter])
            descriptors.extend([FeatureDescriptor(f'Present_{ENEMY_STR}_{g_name}', FeatureType.Boolean)
                                for g_name in self._enemy_filter])
            descriptors = [FeatureDescriptor(lbl, FeatureType.Boolean) for lbl in self.features_labels()]
        if self.config.unit_group_numeric:
            for g_name, group in self._friendly_filter.items():
                num_units = self.config.max_friendly_units
                max_num = sum(num_units[u] if u in num_units else 0 for u in group)
                descriptors.append(FeatureDescriptor(f'Number_{FRIENDLY_STR}_{g_name}',
                                                     FeatureType.Integer, [0, max_num]))
            for g_name, group in self._enemy_filter.items():
                num_units = self.config.max_enemy_units
                max_num = sum(num_units[u] if u in num_units else 0 for u in group)
                descriptors.append(FeatureDescriptor(f'Number_{ENEMY_STR}_{g_name}',
                                                     FeatureType.Integer, [0, max_num]))
        return descriptors

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        def _update_features(filter, unit_types, cat):
            if cat:
                unit_types = list(set(unit_types))  # interested in unit types, not amount
            for group in filter.values():
                if cat:
                    # at least one unit of the group should be on the environment
                    features.append(np.any(np.in1d(group, unit_types, assume_unique=True)))
                else:
                    # count all units of this group
                    features.append(np.sum(np.in1d(unit_types, group, assume_unique=False)))

        # get units for each faction
        alliance = obs['raw_units'][:, 'alliance']
        friendly_unit_types = obs['raw_units'][np.where(alliance == PlayerRelative.SELF)][:, 'unit_type']
        enemy_unit_types = obs['raw_units'][np.where(alliance != PlayerRelative.SELF)][:, 'unit_type']

        # update features
        features = []
        if self.config.unit_group_categorical:
            _update_features(self._friendly_filter, friendly_unit_types, cat=True)
            _update_features(self._enemy_filter, enemy_unit_types, cat=True)

        if self.config.unit_group_numeric:
            _update_features(self._friendly_filter, friendly_unit_types, cat=False)
            _update_features(self._enemy_filter, enemy_unit_types, cat=False)

        return features
