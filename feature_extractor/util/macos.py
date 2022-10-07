from typing import List
from PIL import Image

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'

CG_WINDOW_NUMBER = 'kCGWindowNumber'
CG_WINDOW_OWNER_NAME = 'kCGWindowOwnerName'
CG_WINDOW_NAME = 'kCGWindowName'
CG_WINDOW_ON_SCREEN = 'kCGWindowIsOnscreen'
TITLE_BAR_HEIGHT = 56  # hand-coded, has to be adjusted when new os version comes out.


def get_window_id(name: str, exact_match: bool = False, match_case: bool = True,
                  owner: str = None, on_screen: bool = True) -> List[int]:
    """
    Gets the id of the window with the given name. Only valid in macOS.
    :param str name: the name of the window whose id we want to retrieve.
    :param bool exact_match: whether to perform exact matching.
    :param bool match_case: whether to match the case.
    :param str owner: the name of the window owner whose id we want to retrieve.
    :param bool on_screen: require the window to have the "on screen" flag with a `True` value.
    :rtype: list[int]
    :return: the id of the window or -1 if no window with the given name was found.
    """
    import Quartz.CoreGraphics as CG

    if not match_case:
        name = name.lower()

    def match(target_title, window_title):
        window_title = window_title if match_case else window_title.lower()
        is_on_screen = not on_screen or (CG_WINDOW_ON_SCREEN in window and window[CG_WINDOW_ON_SCREEN])
        return is_on_screen and ((exact_match and target_title) == window_title or
                                 (not exact_match and target_title in window_title))

    # get list of windows, search for name or owner and state and return id
    wl = CG.CGWindowListCopyWindowInfo(CG.kCGWindowListOptionAll, CG.kCGNullWindowID)
    windows = []
    for window in wl:
        if CG_WINDOW_NAME in window and match(name, window[CG_WINDOW_NAME]):
            windows.append(int(window[CG_WINDOW_NUMBER]))
        elif owner is not None and CG_WINDOW_OWNER_NAME in window and match(owner, window[CG_WINDOW_OWNER_NAME]):
            windows.append(int(window[CG_WINDOW_NUMBER]))
    return windows


def get_window_image(window_d: int, crop_title: bool = True) -> Image.Image:
    """
    Gets an image object of the contents of the given window. Only valid in macOS.
    See: https://stackoverflow.com/a/53607100
    See: https://stackoverflow.com/a/22967912
    :param int window_d: the id of the window that we want to capture.
    :param bool crop_title: whether to crop the title bar part of the window.
    :rtype: Image.Image
    :return: the image representation of the given window.
    """
    import Quartz.CoreGraphics as CG

    # get CG image
    cg_img = CG.CGWindowListCreateImage(
        CG.CGRectNull,
        CG.kCGWindowListOptionIncludingWindow,
        window_d,
        CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageBestResolution)

    width = CG.CGImageGetWidth(cg_img)
    height = CG.CGImageGetHeight(cg_img)
    pixel_data = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(cg_img))
    bpr = CG.CGImageGetBytesPerRow(cg_img)

    # create image and crop title
    img = Image.frombuffer('RGBA', (width, height), pixel_data, 'raw', 'BGRA', bpr, 1)
    if crop_title:
        img = img.crop((0, TITLE_BAR_HEIGHT, width, height))
    return img
