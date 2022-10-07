import platform
from . import macos as macos
from . import windows as win32
from PIL import Image

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def get_screenshot(window_title: str = None,
                   exact_match: bool = False,
                   match_case: bool = True,
                   owner: str = None,
                   on_screen: bool = True) -> Image.Image or None:
    """
    Captures a screenshot of the active window whose title matches the given text.
    :param window_title: the text to be matched against the window title.
    :param bool exact_match: whether to perform exact matching.
    :param bool match_case: whether to match the case.
    :param str owner: the name of the window owner whose id we want to retrieve (Mac OS only).
    :param bool on_screen: require the window to have the "on screen" flag with a `True` value (Mac OS only).
    :rtype: Image.Image or None
    :return: an image with the requested client window contents.
    """
    # checks OS and calls methods accordingly
    if platform.system() == 'Windows':
        windows = win32.get_window_id(window_title, exact_match, match_case)
        if len(windows) > 0:
            return win32.get_window_image(windows[0][0], get_client_window=True)
    elif platform.system() == 'Darwin':
        windows = list(macos.get_window_id(window_title, exact_match, match_case, owner, on_screen))
        if len(windows) > 0:
            return macos.get_window_image(windows[0], crop_title=True)

    # could not capture window
    return None


def get_window(window_title: str = None,
               exact_match: bool = False,
               match_case: bool = True,
               owner: str = None,
               on_screen: bool = True) -> int or None:
    """
    Gets the identifier of the active window whose title matches the given text.
    :param window_title: the text to be matched against the window title.
    :param bool exact_match: whether to perform exact matching.
    :param bool match_case: whether to match the case.
    :param str owner: the name of the window owner whose id we want to retrieve (Mac OS only).
    :param bool on_screen: require the window to have the "on screen" flag with a `True` value (Mac OS only).
    :rtype: int or None
    :return: the window identifier.
    """
    # checks OS and calls methods accordingly
    if platform.system() == 'Windows':
        windows = win32.get_window_id(window_title, exact_match, match_case)
        if len(windows) > 0:
            return windows[0][0]
    elif platform.system() == 'Darwin':
        windows = list(macos.get_window_id(window_title, exact_match, match_case, owner, on_screen))
        if len(windows) > 0:
            return windows[0]
    return None  # could not find window


def get_window_image(window_id: int) -> Image.Image or None:
    """
    Captures a screenshot of the active window with the given id.
    :param int window_id: the identifier of the window to be captured.
    :rtype: Image.Image or None
    :return: an image with the requested client window contents.
    """
    # checks OS and calls methods accordingly
    if platform.system() == 'Windows':
        return win32.get_window_image(window_id, get_client_window=True)
    elif platform.system() == 'Darwin':
        return macos.get_window_image(window_id, crop_title=True)
    return None  # could not capture window
