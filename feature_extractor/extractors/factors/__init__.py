import numpy as np
from typing import Dict
from collections import OrderedDict
from pysc2.lib.named_array import NamedNumpyArray, NamedDict
from pysc2.lib.features import PlayerRelative, FeatureUnit
from feature_extractor.config import SpecialFeatureUnit, FeatureExtractorConfig

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def get_units_factor(config: FeatureExtractorConfig,
                     factor: FeatureUnit,
                     op: str,
                     obs: NamedDict,
                     _filter: Dict[str, np.ndarray],
                     alliance: PlayerRelative,
                     negate_alliance: bool = False):
    """
    Gets the sum of a unit factor for friendly and enemy group of units.
    :param FeatureExtractorConfig config: the feature extractor configuration.
    :param FeatureUnit or SpecialFeatureUnit factor: the name of the factor to be retrieved from the pysc2 observation `feature_unit` vector.
    :param str op: the nae of the numpy operation to be performed among the values of all unit factors for a group.
    :param NamedDict obs: the current observation containing the raw features.
    :param OrderedDict[str, np.ndarray] _filter: the friendly unit groups filter.
    :param PlayerRelative alliance: the alliance to which the units belong to.
    :param bool negate_alliance: whether to consider all units *not* belonging to the `alliance`.
    :rtype: dict[str, float or np.ndarray]
    :return: the units factor values for each unit group.
    """
    # fetches relevant feature layers
    alliances = obs['raw_units'][:, 'alliance']
    units = obs['raw_units'][np.where((alliances != alliance) if negate_alliance else alliances == alliance)]

    # gets operation value over units for each faction and group combination
    unit_factors = {}
    for g_name, g_units in _filter.items():
        values = []
        for unit in units:
            if unit['unit_type'] in g_units:
                values.append(get_factor_value(config, factor, unit))
        if len(values) == 0:
            unit_factors[g_name] = None
        else:
            unit_factors[g_name] = eval(f'np.{op}(values)')

    return unit_factors


def get_factor_value(config: FeatureExtractorConfig, factor: FeatureUnit, unit: NamedNumpyArray):
    """
    Gets the unit's value of the specified factor.
    :param FeatureExtractorConfig config: the feature extractor configuration.
    :param FeatureUnit or SpecialFeatureUnit factor: the name of the factor to be retrieved from the unit.
    :param NamedNumpyArray unit: the raw information about the unit.
    :rtype: float
    :return: the unit's factor value.
    """
    unit_type = unit['unit_type']
    return unit[factor] if isinstance(factor, FeatureUnit) \
        else np.sum(config.unit_costs[unit_type]) if factor == SpecialFeatureUnit.total_cost \
        else config.unit_costs[unit_type][0] if factor == SpecialFeatureUnit.mineral_cost \
        else config.unit_costs[unit_type][1] if factor == SpecialFeatureUnit.gas_cost \
        else 0
