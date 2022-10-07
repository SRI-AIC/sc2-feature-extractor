import numpy as np
from typing import List, Union
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from pysc2.lib.named_array import NamedDict
from pysc2.lib.features import PlayerRelative
from feature_extractor.config import NAME_PARAM_STR, VALUE_PARAM_STR, FeatureExtractorConfig
from feature_extractor.extractors import FeatureExtractor, FRIENDLY_STR, ENEMY_STR, DEFAULT_FEATURE_VAL, FeatureType, \
    FeatureDescriptor
from feature_extractor.extractors.factors import get_units_factor

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class ForceFactorsExtractor(FeatureExtractor):
    """
    An extractor that analyzes friendly and enemy groups of units according to some "factor", providing a different
    label according to the "level" or "amount" of that factor.
    """

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self._friendly_filters = [self._convert_unit_filter(ff.friendly_filter) for ff in self.config.force_factors]
        self._enemy_filters = [self._convert_unit_filter(ff.enemy_filter) for ff in self.config.force_factors]

    def features_labels(self) -> List[str]:
        labels = []
        if self.config.force_factor_categorical:
            for i, ff in enumerate(self.config.force_factors):
                for g in self._friendly_filters[i]:
                    labels.append(f'{ff.name}Cat_{FRIENDLY_STR}_{g}')
                for g in self._enemy_filters[i]:
                    labels.append(f'{ff.name}Cat_{ENEMY_STR}_{g}')
        if self.config.force_factor_numeric:
            for i, ff in enumerate(self.config.force_factors):
                for g in self._friendly_filters[i]:
                    labels.append(f'{ff.name}_{FRIENDLY_STR}_{g}')
                for g in self._enemy_filters[i]:
                    labels.append(f'{ff.name}_{ENEMY_STR}_{g}')
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        desc = []
        if self.config.force_factor_categorical:
            for i, ff in enumerate(self.config.force_factors):
                levels = [level[NAME_PARAM_STR] for level in ff.levels]
                for g in self._friendly_filters[i]:
                    desc.append(FeatureDescriptor(f'{ff.name}Cat_{FRIENDLY_STR}_{g}', FeatureType.Categorical, levels))
                for g in self._enemy_filters[i]:
                    desc.append(FeatureDescriptor(f'{ff.name}Cat_{ENEMY_STR}_{g}', FeatureType.Categorical, levels))
        if self.config.force_factor_numeric:
            for i, ff in enumerate(self.config.force_factors):
                for g in self._friendly_filters[i]:
                    desc.append(FeatureDescriptor(f'{ff.name}_{FRIENDLY_STR}_{g}', FeatureType.Real, [0., 1.]))
                for g in self._enemy_filters[i]:
                    desc.append(FeatureDescriptor(f'{ff.name}_{ENEMY_STR}_{g}', FeatureType.Real, [0., 1.]))
        return desc

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        features = []

        def _add_features(cat):
            # gets factors of units for each faction and group combination
            for i, ff in enumerate(self.config.force_factors):

                def _add_groups_features(_filter):
                    # update force factor feature for each group according to the levels' thresholds
                    for g in _filter:
                        feature = DEFAULT_FEATURE_VAL if cat else np.nan
                        if factor_val[g] is not None:
                            if cat:
                                for level in ff.levels:
                                    name = level[NAME_PARAM_STR]
                                    value = level[VALUE_PARAM_STR]
                                    if factor_val[g] <= value:
                                        feature = name
                                        break
                            else:
                                # return ratio between factor value and max level value (between 0 and 1)
                                feature = np.nan if factor_val[g] is None else \
                                    min(1, factor_val[g] / ff.levels[-1][VALUE_PARAM_STR])
                        features.append(feature)

                factor_val = get_units_factor(
                    self.config, ff.factor, ff.op, obs, self._friendly_filters[i], PlayerRelative.SELF)
                _add_groups_features(self._friendly_filters[i])

                factor_val = get_units_factor(
                    self.config, ff.factor, ff.op, obs, self._enemy_filters[i], PlayerRelative.SELF, True)
                _add_groups_features(self._enemy_filters[i])

        if self.config.force_factor_categorical:
            _add_features(cat=True)
        if self.config.force_factor_numeric:
            _add_features(cat=False)

        return features
