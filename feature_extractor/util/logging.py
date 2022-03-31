import logging

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def change_log_handler(log_file: str, level: int = logging.WARN, append: bool = False,
                       fmt: str = '[%(asctime)s %(levelname)s] %(message)s'):
    """
    Changes logger to use the given file.
    :param str log_file: the path to the intended log file.
    :param bool append: whether to append to the log file, if it exists already.
    :param int level: the level of the log messages below which will be saved to file.
    :param str fmt: the formatting string for the messages.
    :return:
    """
    log = logging.getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    file_handler = logging.FileHandler(log_file, 'a' if append else 'w')
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    log.level = file_handler.level = stream_handler.level = level
