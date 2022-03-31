import io
import logging
import re
import tarfile
import tqdm
import numpy as np
import pandas as pd
from typing import List
from feature_extractor.extractors import EPISODE_STR, REPLAY_FILE_STR, TIME_STEP_STR
from feature_extractor.util.io import get_file_name_without_extension

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def merge_feature_files(files: List[str], dtype=None,
                        show_progress: bool = False, use_replay_name: bool = True) -> pd.DataFrame:
    """
    Loads and merges different feature CSV files into a single pandas dataframe.
    :param List[str] files: a list with paths to (possibly zipped) CSV feature files.
    :param dtype: data type(s) for the features.
    :param bool show_progress: whether to show merge progress with tqdm.
    :param bool use_replay_name: whether to get each episode's ID from the replay's file name.
    :rtype: pd.DataFrame
    :return: a pandas dataframe containing all the loaded data.
    """

    def _add_df(df, num_eps):
        eps = df[EPISODE_STR].unique()
        new_eps = set()
        first_feat_idx = df.columns.values.tolist().index(REPLAY_FILE_STR) + 1
        for i in range(first_feat_idx, len(df.columns)):
            if df[df.columns[i]].dtype not in [np.int, np.float]:
                df[df.columns[i]] = df[df.columns[i]].astype(dtype)

        old_to_new = {}
        for i, ep in enumerate(eps):
            new_ep = re.findall(r'ep(\d+)', df[df[EPISODE_STR] == ep][REPLAY_FILE_STR].values[0])
            if use_replay_name and len(new_ep) > 0:
                # extracts episode indices from replay file names
                if new_ep[0] in new_eps:
                    raise ValueError(f'Another episode was already added with the same replay name.')
                else:
                    old_to_new[ep] = int(new_ep[0])
            else:
                old_to_new[ep] = num_eps + i  # use sequential ep index
        df[EPISODE_STR].replace(to_replace=old_to_new, inplace=True)
        dfs.append(df)
        num_eps += len(eps)
        return num_eps

    num_eps = 0
    if show_progress:
        files = tqdm.tqdm(files)
    dfs = []

    for file in files:
        logging.info(f'Loading {file}...')
        if file.endswith('.csv'):
            num_eps = _add_df(pd.read_csv(file), num_eps)
        elif file.endswith('.pkl.gz'):
            num_eps = _add_df(pd.read_pickle(file), num_eps)
        elif file.endswith('.tar.gz'):
            with tarfile.open(file, mode='r:gz') as f:
                for member in f.getmembers():
                    f_ = f.extractfile(member)
                    if f_ is None:
                        continue
                    content = io.BytesIO(f_.read())
                    num_eps = _add_df(pd.read_csv(content), num_eps)
        else:
            raise ValueError(f'Cannot load file: {file}')

    df = pd.concat(dfs, ignore_index=True)  # concat and regenerate index, sort by episode
    df.sort_values([EPISODE_STR, TIME_STEP_STR], inplace=True, ascending=[True, True])
    return df
