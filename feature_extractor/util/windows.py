from typing import Tuple, List
from PIL import Image

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def get_window_id(title_text: str, exact_match: bool = False, match_case: bool = True) -> List[Tuple[int, str]]:
    """
    Gets the handle of all the windows whose title match the given text.
    From: https://stackoverflow.com/a/3278356
    :param str title_text: the text to be matched against the window title.
    :param bool exact_match: whether to perform exact matching.
    :param bool match_case: whether to match the case.
    :rtype: list[(int,str)]
    :return: a list containing tuples with the window handle and title matching the given text.
    """
    import pywintypes  # do not remove
    import win32gui  # pywin32

    def _window_callback(hwnd, all_windows):
        all_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

    if not match_case:
        title_text = title_text.lower()
    all_windows = []
    win32gui.EnumWindows(_window_callback, all_windows)
    windows = []
    for hwnd, title in all_windows:
        title_ = title if match_case else title.lower()
        if exact_match and title_text == title_ or not exact_match and title_text in title_:
            windows.append((hwnd, title))
    return windows


def get_window_image(hwnd: int, get_client_window: bool = True) -> Image.Image or None:
    """
    Gets the image of the window corresponding to the given handle.
    From: https://stackoverflow.com/a/24352388
    :param int hwnd: the handle of the window from which to extract the image. If `None`, gets a screenshot of the whole
    desktop window.
    :param bool get_client_window: whether to extract the client window contents. If `False`, extracts the whole window
    content.
    :rtype: Image.Image
    :return: an image with the requested client window contents.
    """
    import pywintypes  # do not remove
    import win32gui  # pywin32
    import win32ui
    from ctypes import windll

    windll.user32.SetProcessDPIAware()
    if get_client_window:
        left, top, right, bot = win32gui.GetClientRect(hwnd)
    else:
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bitmap)

    if get_client_window:
        result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
    else:
        result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)

    bmp_info = bitmap.GetInfo()
    bmp_str = bitmap.GetBitmapBits(True)

    im = Image.frombuffer('RGB',
                          (bmp_info['bmWidth'], bmp_info['bmHeight']),
                          bmp_str, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    # PrintWindow Succeeded
    return im if result == 1 else None
