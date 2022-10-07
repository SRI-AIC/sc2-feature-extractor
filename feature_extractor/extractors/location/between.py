import numpy as np
from itertools import product
from typing import List, Union
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from pysc2.lib.named_array import NamedDict
from pysc2.lib.features import PlayerRelative
from feature_extractor.config import FeatureExtractorConfig
from feature_extractor.extractors import DEFAULT_FEATURE_VAL, FeatureExtractor, FeatureType, FeatureDescriptor
from feature_extractor.extractors.location import get_unit_locations

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def _is_on_path(points, point1, point2, epsilon=.05):
    """
    Checks whether any of a list of points lies within a straight line connecting two other points.
    :param np.ndarray or list[np.ndarray] points: the points.
    :param np.ndarray point1: the first point defining .
    :param np.ndarray point2: the second point.
    :param float epsilon: the tolerance, in radians, for a point to be considered in the line.
    :rtype: bool
    :return: `True` if any of the given points lies within the line, `False` otherwise.
    """

    def unit_vector(vec):
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[np.where(norm == 0)] = 1
        return vec / norm

    points = np.asarray(points)
    point1 = point1.reshape(1, -1)
    point2 = point2.reshape(1, -1)
    angles = np.arccos(
        np.clip(np.dot(unit_vector(point1 - points), unit_vector(point2 - points).T).diagonal(), -1., 1.))
    return np.any(np.abs(angles - np.pi) < epsilon)


class BetweenExtractor(FeatureExtractor):
    """
    An extractor that detects whether an enemy "barrier" (unit type group) is between every friendly and enemy unit
    within different groups.
    """

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config)
        self._friendly_filter = self._convert_unit_filter(self.config.between_friendly_filter)
        self._enemy_filter = self._convert_unit_filter(self.config.between_enemy_filter)
        self._barrier_filter = self._convert_unit_filter(self.config.between_barrier_filter)
        self._group_combs = list(product(
            self._barrier_filter.keys(), self._friendly_filter.keys(), self._enemy_filter.keys()))

    def features_labels(self) -> List[str]:
        labels = []
        if self.config.between_categorical:
            labels.extend([f'IsBetween_{bg}_{fg}_{eg}' for bg, fg, eg in self._group_combs])
        if self.config.between_numeric:
            labels.extend([f'Between_{bg}_{fg}_{eg}' for bg, fg, eg in self._group_combs])
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        descriptors = []
        if self.config.between_categorical:
            descriptors.extend([FeatureDescriptor(f'IsBetween_{bg}_{fg}_{eg}', FeatureType.Boolean)
                                for bg, fg, eg in self._group_combs])
        if self.config.between_numeric:
            descriptors.extend([FeatureDescriptor(f'Between_{bg}_{fg}_{eg}', FeatureType.Real, [0., 1.])
                                for bg, fg, eg in self._group_combs])
        return descriptors

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        # gets locations of units for each faction and barrier and group combination
        barrier_locs = get_unit_locations(obs, self._barrier_filter, PlayerRelative.SELF, True)
        friendly_locs = get_unit_locations(obs, self._friendly_filter, PlayerRelative.SELF)
        enemy_locs = get_unit_locations(obs, self._enemy_filter, PlayerRelative.SELF, True)

        # update barrier-between feature
        features = []

        def _add_features(cat):
            for bg, fg, eg in self._group_combs:

                # checks for empty groups
                if len(friendly_locs[fg]) == 0 or len(enemy_locs[eg]) == 0 or len(barrier_locs[bg]) == 0:
                    features.append(DEFAULT_FEATURE_VAL if cat else np.nan)
                    continue

                # removes enemy locations from barrier group (to avoid self-obstruction)
                barrier_group_locs = []
                for barrier_loc in barrier_locs[bg]:
                    loc_compare = barrier_loc == enemy_locs[eg]
                    if not np.any(np.logical_and(loc_compare[:, 0], loc_compare[:, 1])):
                        barrier_group_locs.append(barrier_loc)

                # checks for empty group
                if len(barrier_group_locs) == 0:
                    features.append(DEFAULT_FEATURE_VAL if cat else np.nan)
                    continue

                # determines min number of pairs that have to be blocked by a barrier unit
                num_pairs = len(friendly_locs[fg]) * len(enemy_locs[eg])
                min_between_pairs = int(num_pairs * self.config.between_units_ratio)
                between_pairs = 0
                num_unchecked_pairs = num_pairs
                for friendly_loc, enemy_loc in product(friendly_locs[fg], enemy_locs[eg]):

                    # if barrier was found between these 2 units, increase counter
                    if _is_on_path(barrier_group_locs, friendly_loc, enemy_loc, self.config.barrier_angle_threshold):
                        between_pairs += 1

                    if cat:
                        # if categorical/boolean, checks for early termination:
                        # we either have enough between pairs or no way of getting enough between pairs
                        num_unchecked_pairs -= 1
                        if between_pairs >= min_between_pairs or (
                                between_pairs + num_unchecked_pairs) < min_between_pairs:
                            break

                if cat:
                    # the feature is True only if there is a barrier between a minimum number of pairs of units
                    features.append(between_pairs >= min_between_pairs)
                else:
                    # return barrier between ratio
                    features.append(between_pairs / num_pairs)

        if self.config.between_categorical:
            _add_features(cat=True)
        if self.config.between_numeric:
            _add_features(cat=False)

        return features
