import numpy as np
from enum import IntEnum
from typing import List, Union, Dict
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from pysc2.lib.named_array import NamedDict
from pysc2.lib.features import PlayerRelative, FeatureUnit
from feature_extractor.config import FeatureExtractorConfig, OrderConfig
from feature_extractor.extractors.factors import get_units_factor
from feature_extractor.extractors import FeatureExtractor, DEFAULT_FEATURE_VAL, FeatureType, FeatureDescriptor, \
    FRIENDLY_STR, ENEMY_STR

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

MAX_ORDERS = 4


class _OrdersExtractor(FeatureExtractor):
    """
    An extractor that detects whether any unit within a group of friendly or enemy forces is carrying out some behavior
    dictated by a set of possible pysc2 "orders".
    """

    def __init__(self,
                 config: FeatureExtractorConfig,
                 side: str,
                 orders: List[OrderConfig],
                 max_units: Dict[IntEnum, int]):
        """
        Creates a new order extractor.
        :param FeatureExtractorConfig config: the configuration for the feature extractor.
        """
        super().__init__(config)
        self._filters = [self._convert_unit_filter(o.unit_group_filter) for o in orders]
        self.side = side
        self.orders = orders
        self.max_units = max_units

    def features_labels(self) -> List[str]:
        labels = []
        if self.config.orders_categorical:
            labels.extend([f'{o.name}_{self.side}_{g}'
                           for i, o in enumerate(self.orders) for g in self._filters[i]])
        if self.config.orders_numeric:
            labels.extend([f'Number{o.name}_{self.side}_{g}'
                           for i, o in enumerate(self.orders) for g in self._filters[i]])
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        descriptors = []
        if self.config.orders_categorical:
            descriptors.extend([FeatureDescriptor(f'{o.name}_{self.side}_{g}', FeatureType.Boolean)
                                for i, o in enumerate(self.orders) for g in self._filters[i]])
        if self.config.orders_numeric:
            for i, o in enumerate(self.orders):
                for g_name, group in self._filters[i].items():
                    max_num = sum(self.max_units[u] if u in self.max_units else 0 for u in group)
                    descriptors.append(FeatureDescriptor(
                        f'Number{o.name}_{self.side}_{g_name}', FeatureType.Integer, [0, max_num]))
        return descriptors

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        # gets orders of units for each faction and group combination
        features = []

        def _add_features(cat):
            for i, o in enumerate(self.orders):
                unit_orders = [get_units_factor(
                    self.config, FeatureUnit[f'order_id_{j}'], 'array', obs, self._filters[i], PlayerRelative.SELF,
                    self.side != FRIENDLY_STR
                ) for j in range(MAX_ORDERS)]
                order_lens = get_units_factor(
                    self.config, FeatureUnit.order_length, 'array', obs, self._filters[i], PlayerRelative.SELF,
                    self.side != FRIENDLY_STR)

                for g in self._filters[i]:
                    # gets set of orders for units in this group
                    g_orders = np.array([uo[g] for uo in unit_orders])  # shape: (MAX_ORDERS, num_units)
                    if None in g_orders:
                        g_orders = set()  # no units from this group present
                    else:
                        g_orders = [g_orders[:order_lens[g][j], j]
                                    for j in range(g_orders.shape[1])]  # shape: (num_units, unit_orders_len*)

                    # updates feature
                    if cat:
                        # checks whether there's at least one active order on any unit
                        g_orders = list(set().union(*g_orders))
                        features.append(DEFAULT_FEATURE_VAL
                                        if len(g_orders) == 0 or (len(g_orders) == 1 and None in g_orders)
                                        else np.any(np.in1d(g_orders, o.raw_abilities, assume_unique=True)))
                    else:
                        # count number of units executing any of the orders (no duplicates)
                        features.append(np.nan
                                        if all(len(u_orders) == 0 or (len(u_orders) == 1 and None in u_orders)
                                               for u_orders in g_orders)
                                        else np.sum(np.any(np.in1d(u_orders, o.raw_abilities, assume_unique=True))
                                                    for u_orders in g_orders))

        if self.config.orders_categorical:
            _add_features(cat=True)

        if self.config.orders_numeric:
            _add_features(cat=False)

        return features


class FriendlyOrdersExtractor(_OrdersExtractor):
    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config, FRIENDLY_STR, config.friendly_orders, config.max_friendly_units)


class EnemyOrdersExtractor(_OrdersExtractor):
    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config, ENEMY_STR, config.enemy_orders, config.max_enemy_units)
