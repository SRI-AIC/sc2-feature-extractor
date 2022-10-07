import numpy as np
from itertools import product
from typing import List, Union
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from pysc2.lib.named_array import NamedDict
from pysc2.lib.features import PlayerRelative
from feature_extractor.config import FeatureExtractorConfig
from feature_extractor.extractors import FeatureExtractor, DEFAULT_FEATURE_VAL, FeatureType, FeatureDescriptor
from feature_extractor.extractors.factors import get_units_factor

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class ForceRelativeFactorsExtractor(FeatureExtractor):
    """
    An extractor that compares friendly and enemy groups of units according to some "factor", providing a different
    label depending on the "level" or "amount" of that factor.
    """

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self._friendly_filters = [self._convert_unit_filter(ff.friendly_filter)
                                  for ff in self.config.force_relative_factors]
        self._enemy_filters = [self._convert_unit_filter(ff.enemy_filter)
                               for ff in self.config.force_relative_factors]
        self._group_combs = [list(product(friendly_filter.keys(), enemy_filter.keys()))
                             for friendly_filter in self._friendly_filters
                             for enemy_filter in self._enemy_filters]

    def features_labels(self) -> List[str]:
        labels = []
        if self.config.force_relative_categorical:
            for i, ff in enumerate(self.config.force_relative_factors):
                for fg, eg in self._group_combs[i]:
                    labels.append(f'Relative{ff.name}Cat_{fg}_{eg}')
        if self.config.force_relative_numeric:
            for i, ff in enumerate(self.config.force_relative_factors):
                for fg, eg in self._group_combs[i]:
                    labels.append(f'Relative{ff.name}_{fg}_{eg}')
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        desc = []
        if self.config.force_relative_categorical:
            for i, ff in enumerate(self.config.force_relative_factors):
                levels = [DEFAULT_FEATURE_VAL, ff.disadvantage, ff.advantage, ff.balanced]
                for fg, eg in self._group_combs[i]:
                    desc.append(FeatureDescriptor(f'Relative{ff.name}Cat_{fg}_{eg}', FeatureType.Categorical, levels))
        if self.config.force_relative_numeric:
            for i, ff in enumerate(self.config.force_relative_factors):
                for fg, eg in self._group_combs[i]:
                    desc.append(FeatureDescriptor(f'Relative{ff.name}_{fg}_{eg}', FeatureType.Real, [-1., 1.]))
        return desc

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        features = []

        if self.config.force_relative_categorical:
            # gets factors of units for each faction and group combination
            for i, ff in enumerate(self.config.force_relative_factors):
                friendly_factors = get_units_factor(
                    self.config, ff.factor, 'sum', obs, self._friendly_filters[i], PlayerRelative.SELF)
                enemy_factors = get_units_factor(
                    self.config, ff.factor, 'sum', obs, self._enemy_filters[i], PlayerRelative.SELF, True)

                # update army relative factor feature according to threshold
                for fg, eg in self._group_combs[i]:
                    fgf = friendly_factors[fg]
                    egf = enemy_factors[eg]
                    if fgf is None or egf is None:
                        features.append(DEFAULT_FEATURE_VAL)
                        continue

                    ratio = 1 if fgf == 0 and egf == 0 else 1 / ff.ratio if egf == 0 else fgf / egf
                    if ratio < ff.ratio:
                        features.append(ff.disadvantage)
                    elif 1. / ratio < ff.ratio:
                        features.append(ff.advantage)
                    else:
                        features.append(ff.balanced)

        if self.config.force_relative_numeric:
            # gets factors of units for each faction and group combination
            for i, ff in enumerate(self.config.force_relative_factors):
                friendly_factors = get_units_factor(
                    self.config, ff.factor, 'sum', obs, self._friendly_filters[i], PlayerRelative.SELF)
                enemy_factors = get_units_factor(
                    self.config, ff.factor, 'sum', obs, self._enemy_filters[i], PlayerRelative.SELF, True)

                # update army relative factor ratio feature
                for fg, eg in self._group_combs[i]:
                    fgf = friendly_factors[fg]
                    egf = enemy_factors[eg]
                    if fgf is None or egf is None:
                        features.append(np.nan)
                        continue

                    # return normalized ratio in [-1, 1], < 0 if in disadvantage, > 0 if in advantage, 0 if balanced
                    if fgf == egf:
                        features.append(0.)  # equal factor values
                    elif fgf > egf:
                        features.append(1. - egf / fgf)
                    else:
                        features.append(-1. + fgf / egf)

        return features
