import sys
import typing
import numpy as np
from abc import abstractmethod
from collections import OrderedDict
from enum import IntEnum
from typing import List, Dict, Optional, Union
from pysc2.lib.named_array import NamedDict
from s2clientprotocol.sc2api_pb2 import ResponseObservation
from feature_extractor.config import FeatureExtractorConfig

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

ALL_UNITS_GROUP = 'All'

EPISODE_STR = 'Episode'
TIME_STEP_STR = 'Timestep'
REPLAY_FILE_STR = 'File'

FRIENDLY_STR = 'Friendly'
ENEMY_STR = 'Enemy'

DEFAULT_FEATURE_VAL = 'Undefined'  # assigned to unknown values of features (equivalent to having all values = False)


class FeatureType(IntEnum):
    Boolean = 1  # boolean + Undefined
    BooleanPositive = 2  # boolean (true/false)
    Categorical = 3
    String = 4
    Integer = 5
    Real = 6


class FeatureDescriptor(dict):
    """
    An object that describes a high-level feature.
    """

    def __init__(self, name: str, feature_type: FeatureType, feature_values: Optional[List[Union[str, float]]] = None):
        """
        Creates a descriptor for a new feature.
        :param str name: the name of the feature.
        :param FeatureType feature_type: the type of feature.
        :param list[str or float] feature_values: the list of possible values for this feature (for categorical features).
        """
        # for easy conversion to json
        super().__init__(type=feature_type.name,
                         name=name.replace('"', ''),
                         values=feature_values)
        self.name = name
        self.feature_type = feature_type
        self.feature_values = feature_values


class FeatureExtractor(object):
    """
    An interface for feature extractors.
    """

    def __init__(self, config: FeatureExtractorConfig):
        """
        Creates a new feature extractor.
        :param FeatureExtractorConfig config: the configuration for the feature extractor.
        """
        self.config = config

    @abstractmethod
    def features_labels(self) -> List[str]:
        """
        Gets a list of the labels of the features extracted by this extractor.
        :rtype: list[str]
        :return: the list of (ordered) labels of the features extracted by this extractor.
        """
        pass

    @abstractmethod
    def features_descriptors(self) -> List[FeatureDescriptor]:
        """
        Gets a list with the descriptors for each feature extracted by this extractor.
        :rtype: list[FeatureDescriptor]
        :return: this list of (ordered) descriptors of the features extracted by this extractor.
        """
        pass

    @abstractmethod
    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:
        """
        Extracts the features from the given observation.
        :param int ep: the episode that this observation was made.
        :param int step: the episode time-step in which this observation was made.
        :param NamedDict obs: the current observation containing the raw features.
        :param ResponseObservation pb_obs: the observation in protobuf form.
        :rtype: list[bool]
        :return: the list of (ordered) features identified by this extractor.
        """
        pass

    def reset(self, obs: NamedDict, metadata: Optional[Dict] = None):
        """
        Resets the internal state of the feature extractor. Called when the first observation is received.
        :param NamedDict obs: the first observation containing the raw features.
        :param dict metadata: a dictionary containing user defined data for this replay that can be used for feature extraction.
        """
        pass

    def _convert_unit_filter(self, unit_filter: List[str or IntEnum]) -> typing.OrderedDict[str, np.ndarray]:
        """
        Converts a filter into a dictionary of sets of units to facilitate feature extraction.
        :param list[str or IntEnum] unit_filter: the unit filter for an extractor.
        :rtype: OrderedDict[str, np.ndarray]
        :return: a dictionary of sets of units for which we want to separate feature extraction.
        """
        return OrderedDict([(g, np.array([g_unit.value for g_unit in self.config.groups[g]]))
                            if g in self.config.groups else (g.name, np.array([g.value]))
                            for g in unit_filter])


class MetaExtractor(FeatureExtractor):
    """
    An extractor that collects meta information on the replay.
    """

    def __init__(self, config):
        super().__init__(config)
        self.replay_name = ''

    def features_labels(self) -> List[str]:
        return [EPISODE_STR, TIME_STEP_STR, REPLAY_FILE_STR]

    def extract(self, ep: int, step: int, obs: NamedDict, pb_obs: ResponseObservation) -> \
            List[Union[bool, int, float, str]]:
        return [ep, step, self.replay_name]

    def features_descriptors(self) -> List[FeatureDescriptor]:
        return [FeatureDescriptor(EPISODE_STR, FeatureType.Integer, [0, sys.maxsize]),
                FeatureDescriptor(TIME_STEP_STR, FeatureType.Integer, [0, sys.maxsize]),
                FeatureDescriptor(REPLAY_FILE_STR, FeatureType.String)]
