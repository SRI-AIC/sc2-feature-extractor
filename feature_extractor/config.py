import copy
import jsonpickle
from typing import List, Union, Dict, Tuple
from enum import IntEnum
from pysc2.lib.features import FeatureUnit
from pysc2.lib.units import Terran, Zerg, Neutral, Protoss  # needed to parse the unit types

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

NAME_PARAM_STR = 'name'
VALUE_PARAM_STR = 'value'


class SpecialFeatureUnit(IntEnum):
    """
    These correspond to unit factors that can be determined based on the unit type.
    """
    total_cost = 0
    mineral_cost = 1
    gas_cost = 2


_FEATURE_UNIT_NAMES = set(f.name for f in FeatureUnit)
_SPECIAL_FEATURE_UNIT_NAMES = set(f.name for f in SpecialFeatureUnit)


class ForceFactorConfig(object):
    """
    Contains the parameters for the extraction of features related to some "factor" of friendly and enemy units.
    """
    groups: Dict[str, List[IntEnum]] = {}

    def __init__(self,
                 factor: Union[FeatureUnit, SpecialFeatureUnit],
                 name: str,
                 op: str,
                 friendly_filter: List[Union[str, IntEnum]],
                 enemy_filter: List[Union[str, IntEnum]],
                 levels: List[Dict[str, Union[float, int]]]):
        """
        Creates a new factor.
        :param FeatureUnit or SpecialFeatureUnit factor: the name of the factor to be retrieved from the pysc2 observation `feature_unit` vector.
        :param str name: the name of this factor config that will be used to create the name of the features.
        :param str op: the nae of the numpy operation to be performed among the values of all unit factors for a group.
        :param list[str or IntEnum] friendly_filter: the filter for the friendly groups/units to be used in feature extraction.
        :param list[str or IntEnum] enemy_filter: the filter for the friendly groups/units to be used in feature extraction.
        :param list[dict[str,object]] levels: the different levels for the feature factor extraction, each a label and corresponding value.
        """
        self.factor = factor
        self.name = name
        self.op = op
        self.friendly_filter = friendly_filter
        self.enemy_filter = enemy_filter
        self.levels = levels

    def convert_serialize(self):
        config = copy.copy(self)
        del config.groups
        config.factor = self.factor.name
        config.friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.friendly_filter]
        config.enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.enemy_filter]
        return config

    def convert_deserialize(self):
        self.factor = FeatureUnit[self.factor] if self.factor in _FEATURE_UNIT_NAMES \
            else SpecialFeatureUnit[self.factor] if self.factor in _SPECIAL_FEATURE_UNIT_NAMES \
            else FeatureUnit[0]
        self.friendly_filter = [g if g in self.groups else eval(g) for g in self.friendly_filter]
        self.enemy_filter = [g if g in self.groups else eval(g) for g in self.enemy_filter]


class ForceRelativeFactorConfig(object):
    """
    Contains the parameters for the extraction of features related to some "factor" between groups of friendly and enemy units.
    """
    groups: Dict[str, List[IntEnum]] = {}

    def __init__(self,
                 factor: FeatureUnit,
                 name: str,
                 friendly_filter: List[Union[str, IntEnum]],
                 enemy_filter: List[Union[str, IntEnum]],
                 ratio: float,
                 advantage: str,
                 disadvantage: str,
                 balanced: str):
        """
        Creates a new factor.
        :param FeatureUnit factor: the name of the factor to be retrieved from the pysc2 observation `feature_unit` vector.
        :param str name: the name of this factor config that will be used to create the name of the features.
        :param list[str or IntEnum] friendly_filter: the filter for the friendly groups/units to be used in feature extraction.
        :param list[str or IntEnum] enemy_filter: the filter for the friendly groups/units to be used in feature extraction.
        :param float ratio: the threshold for the ratio between the sum of the factor values of friendly and enemy units
        for the friendlies to be considered in minority/disadvantage.
        :param str advantage: the value of the feature when the friendly force is in advantage.
        :param str disadvantage: the value of the feature when the friendly force is in disadvantage.
        :param str balanced: the value of the feature when the forces are balanced.
        """
        self.factor = factor
        self.name = name
        self.friendly_filter = friendly_filter
        self.enemy_filter = enemy_filter
        self.ratio = ratio
        self.advantage = advantage
        self.disadvantage = disadvantage
        self.balanced = balanced

    def convert_serialize(self):
        config = copy.copy(self)
        del config.groups
        config.factor = self.factor.name
        config.friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.friendly_filter]
        config.enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.enemy_filter]
        return config

    def convert_deserialize(self):
        self.factor = FeatureUnit[self.factor] if self.factor in _FEATURE_UNIT_NAMES \
            else SpecialFeatureUnit[self.factor] if self.factor in _SPECIAL_FEATURE_UNIT_NAMES \
            else FeatureUnit[0]
        self.friendly_filter = [g if g in self.groups else eval(g) for g in self.friendly_filter]
        self.enemy_filter = [g if g in self.groups else eval(g) for g in self.enemy_filter]


class OrderConfig(object):
    """
    Contains the parameters for the extraction of features detecting the execution of a set of SC2 orders by groups of
    units.
    """
    groups: Dict[str, List[IntEnum]] = {}

    def __init__(self, name: str, raw_abilities: List[int], unit_group_filter: List[Union[str, IntEnum]]):
        """
        Creates a new order feature configuration
        :param str name: the name of the order feature.
        :param list[int] raw_abilities: list ids corresponding to SC2 raw abilities, as defined in
        pysc2.lib.actions._RAW_FUNCTIONS.
        :param list[str or IntEnum] unit_group_filter: the filter for the groups/units to be used in feature extraction.
        """
        self.name = name
        self.raw_abilities = raw_abilities
        self.unit_group_filter = unit_group_filter

    def convert_serialize(self):
        config = copy.copy(self)
        del config.groups
        config.unit_group_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.unit_group_filter]
        return config

    def convert_deserialize(self):
        self.unit_group_filter = [g if g in self.groups else eval(g) for g in self.unit_group_filter]


class FeatureExtractorConfig(object):
    """
    Contains the parameters of the feature extractor.
    """

    def __init__(self,
                 sample_int: int,
                 friendly_id: int,
                 groups: Dict[str, List[IntEnum]],
                 unit_costs: Dict[IntEnum, List[float]],
                 max_friendly_units: Dict[IntEnum, int],
                 max_enemy_units: Dict[IntEnum, int],
                 unit_group_friendly_filter: List[Union[str, IntEnum]],
                 unit_group_enemy_filter: List[Union[str, IntEnum]],
                 distance_friendly_filter: List[Union[str, IntEnum]],
                 distance_enemy_filter: List[Union[str, IntEnum]],
                 melee_range_ratio: float,
                 close_range_ratio: float,
                 far_range_ratio: float,
                 force_factors: List[ForceFactorConfig],
                 force_relative_factors: List[ForceRelativeFactorConfig],
                 under_attack_friendly_filter: List[Union[str, IntEnum]],
                 under_attack_enemy_filter: List[Union[str, IntEnum]],
                 elevation_friendly_filter: List[Union[str, IntEnum]],
                 elevation_enemy_filter: List[Union[str, IntEnum]],
                 low_elevation: int,
                 medium_elevation: int,
                 high_elevation: int,
                 concentration_friendly_filter: List[Union[str, IntEnum]],
                 concentration_enemy_filter: List[Union[str, IntEnum]],
                 compact_ratio: float,
                 spread_ratio: float,
                 scattered_ratio: float,
                 friendly_move_friendly_filter: List[Union[str, IntEnum]],
                 friendly_move_enemy_filter: List[Union[str, IntEnum]],
                 enemy_move_friendly_filter: List[Union[str, IntEnum]],
                 enemy_move_enemy_filter: List[Union[str, IntEnum]],
                 velocity_threshold: float,
                 max_velocity: float,
                 advance_angle_thresh: Tuple[float, float],
                 retreat_angle_thresh: Tuple[float, float],
                 between_friendly_filter: List[Union[str, IntEnum]],
                 between_enemy_filter: List[Union[str, IntEnum]],
                 between_barrier_filter: List[Union[str, IntEnum]],
                 between_units_ratio: float,
                 barrier_angle_threshold: float,
                 friendly_orders: List[OrderConfig],
                 enemy_orders: List[OrderConfig]):
        """
        Creates a new feature extractor configuration.
        :param int sample_int: the sample interval at which features are extracted.
        :param int friendly_id: the id of the player we consider to be the "friendly" faction. All other players will
        be considered as the "enemy".
        :param dict[str,list[IntEnum]] groups: the logical groups of units used in the feature extractors.
        :param dict[IntEnum, list[float]] unit_costs: the costs associated with each unit type.
        :param dict[IntEnum, int] max_friendly_units: the maximum amount of friendly units per type.
        :param dict[IntEnum, int] max_enemy_units: the maximum amount of enemy units per type.
        :param list[str or IntEnum] unit_group_friendly_filter: the filter for the groups/units used by the unit type extractor.
        :param list[str or IntEnum] unit_group_enemy_filter: the filter for the groups/units used by the unit type extractor.
        :param list[str or IntEnum] distance_friendly_filter: the filter for the groups/units used by the distance extractor.
        :param list[str or IntEnum] distance_enemy_filter: the filter for the groups/units used by the distance extractor.
        :param float melee_range_ratio: the ratio of map size units considered to be at melee range.
        :param float close_range_ratio: the ratio of map size units considered to be at close range.
        :param float far_range_ratio: the ratio of map size units considered to be at far range.
        :param list[ForceFactorConfig] force_factors: the factors configs used by the force factors extractor.
        :param list[ForceRelativeFactorConfig] force_relative_factors: the factor configs used by the relative force factors extractor.
        :param list[str or IntEnum] under_attack_friendly_filter: the filter for the groups/units used by the under attack extractor.
        :param list[str or IntEnum] under_attack_enemy_filter: the filter for the groups/units used by the under attack extractor.
        :param list[str or IntEnum] elevation_friendly_filter: the filter for the groups/units used by the elevation extractor.
        :param list[str or IntEnum] elevation_enemy_filter: the filter for the groups/units used by the elevation extractor.
        :param int low_elevation: the elevation level for a map cell to be considered low.
        :param int medium_elevation: the elevation level for a map cell to be considered medium.
        :param int high_elevation: the elevation level for a map cell to be considered high.
        :param list[str or IntEnum] concentration_friendly_filter: the filter for the groups/units used by the concentration extractor.
        :param list[str or IntEnum] concentration_enemy_filter: the filter for the groups/units used by the concentration extractor.
        :param float compact_ratio: the ratio of map size units for a force to be considered compact.
        :param float spread_ratio: the ratio of map size units for a force to be considered spread.
        :param float scattered_ratio: the ratio of map size units for a force to be considered scattered.
        :param list[str or IntEnum] friendly_move_friendly_filter: the filter for the groups/units used by the movement extractor.
        :param list[str or IntEnum] friendly_move_enemy_filter: the filter for the groups/units used by the movement extractor.
        :param list[str or IntEnum] enemy_move_friendly_filter: the filter for the groups/units used by the movement extractor.
        :param list[str or IntEnum] enemy_move_enemy_filter: the filter for the groups/units used by the movement extractor.
        :param float velocity_threshold: the minimum velocity for a force to be considered moving, measured by the
        amount of spatial cells moved per update step of the force's center of mass.
        :param float max_velocity: the maximum velocity at which units can move.
        :param list[float,float] advance_angle_thresh: the interval for the angle between a forces movement direction
        and a target position for it to be considered as advancing.
        :param list[float,float] retreat_angle_thresh: the interval for the angle between a forces movement direction
        and a target position for it to be considered as retreating.
        :param list[str or IntEnum] between_friendly_filter: the filter for the groups/units used by the between extractor.
        :param list[str or IntEnum] between_enemy_filter: the filter for the groups/units used by the between extractor.
        :param list[str or IntEnum] between_barrier_filter: the filter for the barrier groups/units used by the between extractor.
        :param float between_units_ratio: the ratio of friendly-enemy pairs of units that need to be between the barrier
        for the between feature to be `True`.
        :param float barrier_angle_threshold: the tolerance, in radians, for a barrier unit to be considered within the
        straight line connecting friendly and enemy forces.
        :param list[OrderConfig] friendly_orders: the friendly order features configs used by the orders extractor.
        :param list[OrderConfig] enemy_orders: the enemy order features configs used by the orders extractor.
        """
        self.sample_int = sample_int
        self.friendly_id = friendly_id
        self.groups = groups
        self.unit_costs = unit_costs

        self.max_friendly_units = max_friendly_units
        self.max_enemy_units = max_enemy_units

        self.unit_group_friendly_filter = unit_group_friendly_filter
        self.unit_group_enemy_filter = unit_group_enemy_filter

        self.distance_friendly_filter = distance_friendly_filter
        self.distance_enemy_filter = distance_enemy_filter
        self.melee_range_ratio = melee_range_ratio
        self.close_range_ratio = close_range_ratio
        self.far_range_ratio = far_range_ratio

        self.force_factors = force_factors
        for ff in force_factors:
            ff.groups = groups

        self.force_relative_factors = force_relative_factors
        for ff in force_relative_factors:
            ff.groups = groups

        self.under_attack_friendly_filter = under_attack_friendly_filter
        self.under_attack_enemy_filter = under_attack_enemy_filter

        self.elevation_friendly_filter = elevation_friendly_filter
        self.elevation_enemy_filter = elevation_enemy_filter
        self.low_elevation = low_elevation
        self.medium_elevation = medium_elevation
        self.high_elevation = high_elevation

        self.concentration_friendly_filter = concentration_friendly_filter
        self.concentration_enemy_filter = concentration_enemy_filter
        self.compact_ratio = compact_ratio
        self.spread_ratio = spread_ratio
        self.scattered_ratio = scattered_ratio

        self.friendly_move_friendly_filter = friendly_move_friendly_filter
        self.friendly_move_enemy_filter = friendly_move_enemy_filter
        self.enemy_move_friendly_filter = enemy_move_friendly_filter
        self.enemy_move_enemy_filter = enemy_move_enemy_filter
        self.velocity_threshold = velocity_threshold
        self.max_velocity = max_velocity
        self.advance_angle_thresh = advance_angle_thresh
        self.retreat_angle_thresh = retreat_angle_thresh

        self.between_friendly_filter = between_friendly_filter
        self.between_enemy_filter = between_enemy_filter
        self.between_barrier_filter = between_barrier_filter
        self.between_units_ratio = between_units_ratio
        self.barrier_angle_threshold = barrier_angle_threshold

        self.friendly_orders = friendly_orders
        for o in friendly_orders:
            o.groups = groups
        self.enemy_orders = enemy_orders
        for o in enemy_orders:
            o.groups = groups

    def save_json(self, json_file_path):
        """
        Saves a text file representing this config in a JSON format.
        :param str json_file_path: the path to the JSON file in which to save this config.
        :return:
        """
        jsonpickle.set_preferred_backend('json')
        jsonpickle.set_encoder_options('json', indent=4, sort_keys=False)
        with open(json_file_path, 'w') as json_file:
            json_str = jsonpickle.encode(self.convert_serialize())
            json_file.write(json_str)

    def convert_serialize(self):
        # transforms lists of IntEnum types into corresponding string names
        config = copy.copy(self)
        config.groups = {g_name: [str(unit) for unit in g_units] for g_name, g_units in self.groups.items()}
        config.unit_costs = {str(u_type): costs for u_type, costs in config.unit_costs.items()}
        config.max_friendly_units = {str(u_type): n for u_type, n in config.max_friendly_units.items()}
        config.max_enemy_units = {str(u_type): n for u_type, n in config.max_enemy_units.items()}
        config.force_factors = [ff.convert_serialize() for ff in self.force_factors]
        config.force_relative_factors = [ff.convert_serialize() for ff in self.force_relative_factors]

        config.unit_group_friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                             self.unit_group_friendly_filter]
        config.unit_group_enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                          self.unit_group_enemy_filter]

        config.distance_friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                           self.distance_friendly_filter]
        config.distance_enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.distance_enemy_filter]

        config.concentration_friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                                self.concentration_friendly_filter]
        config.concentration_enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                             self.concentration_enemy_filter]

        config.under_attack_friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                               self.under_attack_friendly_filter]
        config.under_attack_enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                            self.under_attack_enemy_filter]

        config.elevation_friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                            self.elevation_friendly_filter]
        config.elevation_enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.elevation_enemy_filter]

        config.friendly_move_friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                                self.friendly_move_friendly_filter]
        config.friendly_move_enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                             self.friendly_move_enemy_filter]
        config.enemy_move_friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                             self.enemy_move_friendly_filter]
        config.enemy_move_enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in
                                          self.enemy_move_enemy_filter]

        config.between_friendly_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.between_friendly_filter]
        config.between_enemy_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.between_enemy_filter]
        config.between_barrier_filter = [str(g) if isinstance(g, IntEnum) else g for g in self.between_barrier_filter]

        config.friendly_orders = [o.convert_serialize() for o in self.friendly_orders]
        config.enemy_orders = [o.convert_serialize() for o in self.enemy_orders]

        return config

    @staticmethod
    def load_json(json_file_path):
        """
        Loads a config object from the given JSON formatted file.
        :param str json_file_path: the path to the JSON file from which to load a config.
        :rtype: FeatureExtractorConfig
        :return: the config object stored in the given JSON file.
        """
        with open(json_file_path) as json_file:
            conf = jsonpickle.decode(json_file.read())
            conf.convert_deserialize()
            return conf

    def convert_deserialize(self):
        # transforms lists of strings into corresponding IntEnum types
        self.groups = {g_name: [eval(unit) for unit in g_units] for g_name, g_units in self.groups.items()}
        self.unit_costs = {eval(u_type): costs for u_type, costs in self.unit_costs.items()}
        self.max_friendly_units = {eval(u_type): n for u_type, n in self.max_friendly_units.items()}
        self.max_enemy_units = {eval(u_type): n for u_type, n in self.max_enemy_units.items()}

        for ff in self.force_factors:
            ff.groups = self.groups
            ff.convert_deserialize()
        for ff in self.force_relative_factors:
            ff.groups = self.groups
            ff.convert_deserialize()

        self.unit_group_friendly_filter = [g if g in self.groups else eval(g) for g in self.unit_group_friendly_filter]
        self.unit_group_enemy_filter = [g if g in self.groups else eval(g) for g in self.unit_group_enemy_filter]

        self.distance_friendly_filter = [g if g in self.groups else eval(g) for g in self.distance_friendly_filter]
        self.distance_enemy_filter = [g if g in self.groups else eval(g) for g in self.distance_enemy_filter]

        self.concentration_friendly_filter = [g if g in self.groups else eval(g) for g in
                                              self.concentration_friendly_filter]
        self.concentration_enemy_filter = [g if g in self.groups else eval(g) for g in
                                           self.concentration_enemy_filter]

        self.under_attack_friendly_filter = [g if g in self.groups else eval(g) for g in
                                             self.under_attack_friendly_filter]
        self.under_attack_enemy_filter = [g if g in self.groups else eval(g) for g in
                                          self.under_attack_enemy_filter]

        self.elevation_friendly_filter = [g if g in self.groups else eval(g) for g in self.elevation_friendly_filter]
        self.elevation_enemy_filter = [g if g in self.groups else eval(g) for g in self.elevation_enemy_filter]

        self.friendly_move_friendly_filter = [g if g in self.groups else eval(g) for g in
                                              self.friendly_move_friendly_filter]
        self.friendly_move_enemy_filter = [g if g in self.groups else eval(g)
                                           for g in self.friendly_move_enemy_filter]
        self.enemy_move_friendly_filter = [g if g in self.groups else eval(g) for g in
                                           self.enemy_move_friendly_filter]
        self.enemy_move_enemy_filter = [g if g in self.groups else eval(g)
                                        for g in self.enemy_move_enemy_filter]

        self.between_friendly_filter = [g if g in self.groups else eval(g) for g in self.between_friendly_filter]
        self.between_enemy_filter = [g if g in self.groups else eval(g) for g in self.between_enemy_filter]
        self.between_barrier_filter = [g if g in self.groups else eval(g) for g in self.between_barrier_filter]

        for o in self.friendly_orders:
            o.groups = self.groups
            o.convert_deserialize()
        for o in self.enemy_orders:
            o.groups = self.groups
            o.convert_deserialize()
