import sys
import itertools as it
import multiprocessing as mp
import multiprocessing.pool as mpp
from typing import Tuple, Callable

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def _istarmap(self, func, iterable, chunksize=1):
    """
    Starmap-version of imap, see: https://stackoverflow.com/a/57364423/16031961
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


def _istarmap38(self, func, iterable, chunksize=1):
    """
    Starmap-version of imap, Python 3.8+ see: https://stackoverflow.com/a/57364423/16031961
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


# replace function
if sys.version_info[1] >= 8:
    mpp.Pool.istarmap = _istarmap38
else:
    mpp.Pool.istarmap = _istarmap


def get_pool_and_map(processes: int or None, star: bool = False, iterator: bool = False) -> \
        Tuple[mp.Pool or None, Callable]:
    """
    Returns a process pool and mapping function, or a single-process mapping function, depending on the number of
    requested processes.
    :param int processes: number of processes to use. `None` indicates all cores available, `1` uses single process.
    :param bool star: whether to return a starmap function instead of single-argument map.
    :param bool iterator: whether to return the iterator version when multiprocessing is used.
    :rtype: Tuple[mp.Pool, Callable]
    :return: a tuple (pool, mapping_func) containing the pool object (or `None`) and mapping function.
    """
    # selects mapping function according to number of requested processes
    if processes == 1:
        pool = None
        map_func = it.starmap if star else map
    else:
        pool = mp.Pool(processes)
        if iterator:
            map_func = pool.istarmap if star else pool.imap
        else:
            map_func = pool.starmap if star else pool.map
    return pool, map_func
