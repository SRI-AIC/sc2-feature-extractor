import numpy as np
from typing import List, Dict, Optional, Union
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from pysc2.lib.named_array import NamedDict
from pysc2.lib.features import FeatureUnit, PlayerRelative
from feature_extractor.config import FeatureExtractorConfig
from feature_extractor.extractors import FeatureExtractor, FRIENDLY_STR, ENEMY_STR, DEFAULT_FEATURE_VAL, FeatureType, \
    FeatureDescriptor
from feature_extractor.extractors.factors import get_units_factor

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class UnderAttackExtractor(FeatureExtractor):
    """
    An extractor that detects whether friendly and enemy groups of units are under attack by monitoring whether the
    sum of their health is decreasing between consecutive timesteps.
    """

    def __init__(self, config: FeatureExtractorConfig, categorical: bool):
        super().__init__(config)
        self._categorical = categorical
        self._friendly_filter = self._convert_unit_filter(self.config.under_attack_friendly_filter)
        self._enemy_filter = self._convert_unit_filter(self.config.under_attack_enemy_filter)
        self._first_obs = True
        self._prev_friendly_health = {g: 0. for g in self._friendly_filter}
        self._prev_enemy_health = {g: 0. for g in self._enemy_filter}

    def reset(self, obs: NamedDict, metadata: Optional[Dict] = None):
        self._first_obs = True
        self._prev_friendly_health = {g: 0. for g in self._friendly_filter}
        self._prev_enemy_health = {g: 0. for g in self._enemy_filter}

    def features_labels(self) -> List[str]:
        labels = []
        for g in self._friendly_filter:
            labels.append(f'UnderAttack_{FRIENDLY_STR}_{g}')
        for g in self._enemy_filter:
            labels.append(f'UnderAttack_{ENEMY_STR}_{g}')
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        if self._categorical:
            return [FeatureDescriptor(lbl, FeatureType.Boolean) for lbl in self.features_labels()]
        else:
            return [FeatureDescriptor(lbl, FeatureType.Real, [np.finfo(np.float).min, np.finfo(np.float).max])
                    for lbl in self.features_labels()]

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        # get unit groups' health for each faction
        friendly_health = get_units_factor(
            self.config, FeatureUnit.health, 'sum', obs, self._friendly_filter, PlayerRelative.SELF)
        enemy_health = get_units_factor(
            self.config, FeatureUnit.health, 'sum', obs, self._enemy_filter, PlayerRelative.SELF, True)

        # updates under attack feature for each unit group and faction
        features = []
        for g_filter, health, prev_health in \
                [(self._friendly_filter, friendly_health, self._prev_friendly_health),
                 (self._enemy_filter, enemy_health, self._prev_enemy_health)]:
            for g in g_filter:
                if health[g] is None or prev_health[g] is None:
                    features.append(DEFAULT_FEATURE_VAL if self._categorical else np.nan)
                else:
                    if self._categorical:
                        # return whether agent is losing health
                        features.append(not self._first_obs and health[g] < prev_health[g])
                    else:
                        # return difference in health
                        features.append(0. if self._first_obs else health[g] - prev_health[g])
                prev_health[g] = health[g]

        self._first_obs = False

        return features
