import io
import logging
import os
import tarfile
import time
import pandas as pd
import tqdm
from .io import get_file_name_without_extension

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def save_separate_csv_gzip(df: pd.DataFrame,
                           file_path: str,
                           group_by: str,
                           use_group_filename: bool = True,
                           use_tqdm: bool = True):
    """
    Saves a Pandas dataframe to a Gzipped file, saving individual CSV files inside by grouping the data according to
    some column.
    :param pd.DataFrame df: the dataframe to be saved.
    :param str file_path: the path to the gzip archive file in which to save the CSV files.
    :param str group_by: the name of the column used to split the dataframe and create the individual CSV files.
    :param bool use_group_filename: if `True`, uses the `group_by` column values to name the individual CSV files,
    otherwise use sequential number file names (i.e., 0.csv, 1.csv, ...).
    :param bool use_tqdm: whether to use tqdm when splitting/saving the CSV files into the Gzip archive.
    """

    def _get_filename(group: str):
        if os.sep in group or '.' in group:
            return get_file_name_without_extension(group)
        return group

    # split dataframe by group_by column
    num_groups = len(df[group_by].unique())
    logging.info(f'Saving data for {num_groups} groups ("{group_by}") in separate CSV files, '
                 f'compressing them to gzip file:\n\t{file_path}')
    groups = enumerate(df.groupby(group_by))

    # splits data and saves individual CSV files inside a Gzip archive
    with tarfile.open(file_path, mode='w:gz') as fp:
        for i, (g, g_df) in (tqdm.tqdm(groups, total=num_groups) if use_tqdm else groups):
            file_name = _get_filename(g) if use_group_filename else str(i)
            buf = io.BytesIO()
            g_df.to_csv(buf, index=False)
            buf.seek(0)
            tarinfo = tarfile.TarInfo(f'{file_name}.csv')
            tarinfo.mtime = time.time()
            tarinfo.size = len(buf.getvalue())  # have to provide buffer size
            fp.addfile(tarinfo, buf)
