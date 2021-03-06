import argparse
import logging
from enum import IntEnum
from interestingness_xdrl.util.io import save_dict_json

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def save_args(args: argparse.Namespace, file_path: str):
    """
    Saves the given `argparse` arguments to a json file.
    :param Namespace args: the arguments generated by `argparse` to save.
    :param str file_path: the path to the json file where to save the arguments.
    """
    args_dict = vars(args)
    args_dict = {k: v.name if isinstance(v, IntEnum) else v for k, v in args_dict.items()}
    save_dict_json(args_dict, file_path)


def str2bool(v: str) -> bool:
    """
    Converts the given string parsed using `argparse` to a boolean value.
    :param str v: the argument value to be converted.
    :rtype:bool
    :return: the boolean value corresponding to the given argument.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2log_level(v: str or int) -> int:
    """
    Converts the given string parsed using `argparse` to a logging level value. Can either be 0, 1, or 2, corresponding
    to logging levels `WARN`, `INFO` or `DEBUG`, respectively, the string values of correct log levels, or an int
    representing a custom log level.
    :param str or int v: the argument value to be converted.
    :rtype:int
    :return: the log level corresponding to the given argument.
    """
    try:
        v = int(v)
    except ValueError:
        pass
    if isinstance(v, str):
        v = logging.getLevelName(v.upper())
        if isinstance(v, int):
            return v
    if isinstance(v, int):
        return logging.WARN if v == 0 else logging.INFO if v == 1 else logging.DEBUG if v == 2 else v

    raise argparse.ArgumentTypeError('Valid log level expected.')
