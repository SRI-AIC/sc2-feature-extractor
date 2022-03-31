import numpy as np
from enum import IntEnum
from typing import Dict
from pysc2.lib.features import PlayerRelative
from pysc2.lib.named_array import NamedDict

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def get_unit_locations(obs: NamedDict,
                       unit_filter: Dict[str, np.ndarray],
                       alliance: PlayerRelative,
                       negate_alliance: bool = False,
                       raw_units: bool = True) -> Dict[str, np.ndarray]:
    """
    Gets the current locations of the units in the given groups.
    :param NamedDict obs: the current observation containing the raw features.
    :param Dict[str, set[IntEnum]] unit_filter: the unit groups filter.
    :param PlayerRelative alliance: the alliance to which the units belong to.
    :param bool negate_alliance: whether to consider all units *not* belonging to the `alliance`.
    :param bool raw_units: whether to use the "raw_units" array instead of "feature_units".
    :rtype: Dict[str, np.ndarray]
    :return: the locations of the units organized by unit group.
    """
    # fetches relevant feature layers
    units_obs = obs['raw_units'] if raw_units else obs['feature_units']
    alliances = units_obs[:, 'alliance']
    units = units_obs[np.where((alliances != alliance) if negate_alliance else alliances == alliance)]

    # gets locations of units for this faction and each group combination
    loc_idxs = [units._index_names[1]['x'], units._index_names[1]['y']]
    locs = {g_name: np.asarray(units[np.where(np.in1d(units[:, 'unit_type'], g_units))][:, loc_idxs])
            for g_name, g_units in unit_filter.items()}

    return locs


def get_locations_by_unit(obs: NamedDict,
                          unit_filter: Dict[str, np.ndarray],
                          alliance: PlayerRelative,
                          negate_alliance: bool = False,
                          raw_units: bool = True) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Gets the current locations of the units in the given groups, organized by unit "tag".
    :param NamedDict obs: the current observation containing the raw features.
    :param Dict[str, set[IntEnum]] unit_filter: the unit groups filter.
    :param PlayerRelative alliance: the alliance to which the units belong to.
    :param bool negate_alliance: whether to consider all units *not* belonging to the `alliance`.
    :param bool raw_units: whether to use the "raw_units" array instead of "feature_units".
    :rtype: Dict[str, np.ndarray]
    :return: the locations of the units organized by unit group.
    """
    # fetches relevant feature layers
    units_obs = obs['raw_units'] if raw_units else obs['feature_units']
    alliances = units_obs[:, 'alliance']
    units = units_obs[np.where((alliances != alliance) if negate_alliance else alliances == alliance)]

    # gets locations of units for this faction and each group combination
    loc_idxs = [units._index_names[1]['x'], units._index_names[1]['y']]
    tag_idx = units._index_names[1]['tag']
    locs = {}
    for g_name, g_units in unit_filter.items():
        g_units_idxs = np.where(np.in1d(units[:, 'unit_type'], g_units))
        locs[g_name] = {u[tag_idx]: np.asarray(u[loc_idxs]) for u in units[g_units_idxs]}

    return locs
