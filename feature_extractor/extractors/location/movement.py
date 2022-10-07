import numpy as np
from typing import List, Dict, Union, Optional
from itertools import product
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from pysc2.lib.named_array import NamedDict
from pysc2.lib.features import PlayerRelative
from feature_extractor.config import FeatureExtractorConfig
from feature_extractor.extractors import FeatureExtractor, FRIENDLY_STR, ENEMY_STR, DEFAULT_FEATURE_VAL, FeatureType, \
    FeatureDescriptor
from feature_extractor.extractors.location import get_locations_by_unit

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class _RelativeMovementExtractor(FeatureExtractor):
    """
    An extractor that detects the movement of groups of friendly and enemy forces relative to each other, i.e,
    either advancing or retreating.
    """

    def __init__(self, config, force_label, own_filter, other_filter, alliance_self):
        super().__init__(config)
        self._force_label = force_label
        self._alliance_self = alliance_self

        self._own_filter = self._convert_unit_filter(own_filter)
        self._other_filter = self._convert_unit_filter(other_filter)
        self._group_combs = list(product(self._own_filter.keys(), self._other_filter.keys()))

        self._prev_step = 0
        self._prev_own_units = {g: None for g in self._own_filter}
        self._prev_other_units = {g: None for g in self._other_filter}
        self._own_centers = {g: None for g in self._own_filter}
        self._other_centers = {g: None for g in self._other_filter}
        self._own_speeds = {g: np.array([0, 0]) for g in self._own_filter}
        self._other_speeds = {g: np.array([0, 0]) for g in self._other_filter}

    def reset(self, obs: NamedDict, metadata: Optional[Dict] = None):
        self._prev_step = 0
        self._prev_own_units = {g: None for g in self._own_filter}
        self._prev_other_units = {g: None for g in self._other_filter}
        self._own_centers = {g: None for g in self._own_filter}
        self._other_centers = {g: None for g in self._other_filter}
        self._own_speeds = {g: np.array([0, 0]) for g in self._own_filter}
        self._other_speeds = {g: np.array([0, 0]) for g in self._other_filter}

    def features_labels(self) -> List[str]:
        labels = []
        if self.config.movement_categorical:
            for own_g, other_g in self._group_combs:
                labels.append(f'Advancing_{self._force_label}_{own_g}_{other_g}')
                labels.append(f'Retreating_{self._force_label}_{own_g}_{other_g}')
        if self.config.movement_numeric:
            for own_g, other_g in self._group_combs:
                labels.append(f'Velocity_{self._force_label}_{own_g}_{other_g}')
                labels.append(f'Angle_{self._force_label}_{own_g}_{other_g}')
        return labels

    def features_descriptors(self) -> List[FeatureDescriptor]:
        descriptors = []
        if self.config.movement_categorical:
            for own_g, other_g in self._group_combs:
                descriptors.append(FeatureDescriptor(
                    f'Advancing_{self._force_label}_{own_g}_{other_g}', FeatureType.Boolean))
                descriptors.append(FeatureDescriptor(
                    f'Retreating_{self._force_label}_{own_g}_{other_g}', FeatureType.Boolean))
        if self.config.movement_numeric:
            for own_g, other_g in self._group_combs:
                descriptors.append(FeatureDescriptor(
                    f'Velocity_{self._force_label}_{own_g}_{other_g}', FeatureType.Real, [0., 1.]))
                descriptors.append(FeatureDescriptor(
                    f'Angle_{self._force_label}_{own_g}_{other_g}', FeatureType.Real, [0., np.pi]))
        return descriptors

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:

        def get_center_mass(units_locs):
            return None if len(units_locs) == 0 else np.mean(units_locs, axis=0)

        def get_velocity(prev_loc, cur_loc):
            return (cur_loc - prev_loc) / steps if cur_loc is not None and prev_loc is not None else None

        def unit_vector(vec):
            norm = np.linalg.norm(vec)
            return vec if norm == 0 else vec / norm

        def update_force_properties(_filter, alliance, negate_alliance, prev_locs, centers, speeds):
            locs = get_locations_by_unit(obs, _filter, alliance, negate_alliance)
            for g in _filter:
                speeds[g] = None
                centers[g] = None
                if prev_locs[g] is not None:
                    # get intersection of units from last step, compute centers of mass only for those
                    unit_intersect = set(prev_locs[g].keys()).intersection(locs[g].keys())
                    prev_center = get_center_mass(np.array([prev_locs[g][u] for u in unit_intersect]))
                    cur_center = get_center_mass(np.array([locs[g][u] for u in unit_intersect]))
                    speeds[g] = get_velocity(prev_center, cur_center)
                    centers[g] = cur_center
                prev_locs[g] = locs[g]

        # gets locations of units for each group and faction and update velocities
        steps = step - self._prev_step
        update_force_properties(self._own_filter, PlayerRelative.SELF, not self._alliance_self,
                                self._prev_own_units, self._own_centers, self._own_speeds)
        update_force_properties(self._other_filter, PlayerRelative.SELF, self._alliance_self,
                                self._prev_other_units, self._other_centers, self._other_speeds)

        # updates movement features
        features = []
        if self.config.movement_categorical:
            for own_g, other_g in self._group_combs:
                speed = self._own_speeds[own_g]
                center = self._own_centers[own_g]
                target = self._other_centers[other_g]
                if speed is None or target is None:
                    features.extend([DEFAULT_FEATURE_VAL, DEFAULT_FEATURE_VAL]) # no units of one or both sides
                    continue

                # calculate relative direction
                abs_speed = np.linalg.norm(speed)
                angle_to_target = np.arccos(
                    np.clip(np.dot(unit_vector(speed), unit_vector(target - center)), -1., 1.))
                if abs_speed <= self.config.velocity_threshold:
                    features.extend([False, False])  # not enough speed
                elif self.config.advance_angle_thresh[0] <= angle_to_target < self.config.advance_angle_thresh[1]:
                    features.extend([True, False])  # advancing towards target
                elif self.config.retreat_angle_thresh[0] < angle_to_target <= self.config.retreat_angle_thresh[1]:
                    features.extend([False, True])  # retreating from target
                else:
                    features.extend([False, False])  # force is moving, but not relative to target

        if self.config.movement_numeric:
            for own_g, other_g in self._group_combs:
                speed = self._own_speeds[own_g]
                center = self._own_centers[own_g]
                target = self._other_centers[other_g]
                if speed is None or target is None:
                    features.extend([np.nan, np.nan])  # no units of one of the sides
                    continue

                # calculate relative direction, return relative velocity, angle
                abs_speed = np.linalg.norm(speed)
                angle_to_target = np.arccos(
                    np.clip(np.dot(unit_vector(speed), unit_vector(target - center)), -1., 1.))
                features.extend([min(1, abs_speed / self.config.max_velocity), angle_to_target])

        self._prev_step = step
        return features


class FriendlyRelativeMovementExtractor(_RelativeMovementExtractor):
    """
    Captures relative movement for friendly units in relation to enemy units.
    """

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config, FRIENDLY_STR,
                         config.friendly_move_friendly_filter, config.friendly_move_enemy_filter,
                         alliance_self=True)


class EnemyRelativeMovementExtractor(_RelativeMovementExtractor):
    """
    Captures relative movement for enemy units in relation to friendly units.
    """

    def __init__(self, config: FeatureExtractorConfig):
        super().__init__(config, ENEMY_STR,
                         config.enemy_move_enemy_filter, config.enemy_move_friendly_filter,
                         alliance_self=False)
