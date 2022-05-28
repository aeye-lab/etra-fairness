import os
from typing import Dict
from typing import List

import numpy as np
import pandas as pd


# get all csv files recursively
def get_csvs(start_dir: str) -> List[str]:
    return_list = []
    list_dir = os.listdir(start_dir)
    for i in range(len(list_dir)):
        cur_dir = start_dir + '/' + list_dir[i]
        if os.path.isdir(cur_dir):
            cur_dir += '/'
            return_list += get_csvs(cur_dir)
        else:
            if cur_dir.endswith('.csv'):
                return_list.append(cur_dir)
    return return_list


def get_lohr_csvs(start_dir: str) -> List[str]:
    return_list = []
    list_dir = os.listdir(start_dir)
    for i in range(len(list_dir)):
        cur_dir = start_dir + '/' + list_dir[i]
        if os.path.isdir(cur_dir):
            cur_dir += '/'
            return_list += get_lohr_csvs(cur_dir)
        else:
            if (
                cur_dir.endswith('Raw_Sac_Features.csv') or
                cur_dir.endswith('Raw_PSO_Features.csv') or
                cur_dir.endswith('Raw_Fix_Features.csv')
            ):
                return_list.append(cur_dir)
    return return_list


def transform_to_new_seqlen_length(
    X: np.array, new_seq_len: int, skip_padded: bool = False,
) -> np.array:
    """
    Example: if old seq len was 7700, new_seq_len=1000:
    Input X has: 144 x 7700 x n_channels
    Output X has: 144*8 x 1000 x n_channels
    The last piece of each trial 7000-7700 gets padded with first 300 of this piece to be 1000 long
    :param X:
    :param new_seq_len:
    :return:
    """
    n, rest = np.divmod(X.shape[1], new_seq_len)

    if rest > 0 and not skip_padded:
        n_rows = X.shape[0]*(n+1)
    else:
        n_rows = X.shape[0]*n

    X_new = np.nan * np.ones((n_rows, new_seq_len, X.shape[2]))

    idx = 0
    for t in range(0, X.shape[0]):
        for i in range(0, n):
            # cut out 1000 ms piece of trial t
            X_tmp = np.expand_dims(
                X[t, i*new_seq_len: (i+1)*new_seq_len, :], axis=0,
            )

            # concatenate pieces
            X_new[idx, :, :] = X_tmp

            idx = idx + 1

        if rest > 0 and not skip_padded:
            # concatenate last one with pad
            start_idx_last_piece = new_seq_len*(n)
            len_pad_to_add = new_seq_len-rest
            # piece to pad:
            X_incomplete = np.expand_dims(
                X[t, start_idx_last_piece:X.shape[1], :], axis=0,
            )
            # padding piece:
            start_idx_last_piece = new_seq_len*(n-1)
            X_pad = np.expand_dims(
                X[
                    t, start_idx_last_piece:start_idx_last_piece +
                    len_pad_to_add, :
                ], axis=0,
            )

            X_tmp = np.concatenate((X_incomplete, X_pad), axis=1)

            # concatenate last piece of original row t
            X_new[idx, :, :] = X_tmp

            idx = idx + 1

    assert np.sum(
        np.isnan(
            X_new[:, :, 0],
        ),
    ) == 0, 'Cutting into pieces failed, did not fill each position of new matrix.'

    return X_new


def get_data_for_user(
    data_path: str, max_vel: float = 500, delete_nans: bool = True,
    sampling_rate: int = 1000, smooth: bool = True,
    delete_high_velocities: bool = False,
):
    cur_data = pd.read_csv(data_path)
    X = np.array([
        cur_data['x'],
        cur_data['y'],
    ]).T
    X[np.array(cur_data['val']) != 0, :] = np.nan
    if delete_nans:
        not_nan_ids = np.logical_and(~np.isnan(X[:, 0]), ~np.isnan(X[:, 1]))
        X = X[not_nan_ids, :]

    # transform to velocities
    vel_left = vecvel(X, sampling_rate, smooth=smooth)
    vel_left[vel_left > max_vel] = max_vel
    vel_left[vel_left < -max_vel] = -max_vel
    if delete_high_velocities:
        not_high_velocity_ids = np.logical_or(
            np.abs(vel_left[:, 0]) >= max_vel,
            np.abs(vel_left[:, 1]) >= max_vel,
        )
        X = X[not_high_velocity_ids]
        vel_left = vel_left[not_high_velocity_ids]
    X_vel = vel_left

    X_vel_transformed = transform_to_new_seqlen_length(
        X=np.reshape(X_vel, [1, X_vel.shape[0], X_vel.shape[1]]),
        new_seq_len=sampling_rate,
        skip_padded=True,
    )

    user_dict = {
        'X': X,
        'X_deg': X,
        'X_vel': X_vel,
        'path': data_path,
        'X_vel_transformed': X_vel_transformed,
    }
    return user_dict


def get_data_for_user_lohr(data_path: str) -> Dict[str, pd.DataFrame]:
    cur_data = pd.read_csv(data_path)
    user_dict = {
        'X': cur_data,
    }
    return user_dict


# Compute velocity times series from 2D position data
# adapted from Engbert et al.  Microsaccade Toolbox 0.9
# x: array of shape (N,2)
#    yaw and pitch screen or visual angle coordinates, N samples in *chronological* order)
# returns velocity in deg/sec or pix/sec
def vecvel(
    x: np.array, sampling_rate: int = 1000, smooth: bool = True, sample_diff: bool = False,
) -> np.array:
    assert np.array_equal(np.isnan(x[:, 0]), np.isnan(x[:, 1]))
    N = x.shape[0]
    v = np.zeros((N, 2))
    if smooth:  # v based on mean of preceding 2 samples and mean of following 2 samples
        v[2:N-2, :] = (sampling_rate/6)*(
            x[4:N, :] +
            x[3:N-1, :] - x[1:N-3, :] - x[0:N-4, :]
        )
        v[1, :] = (sampling_rate/2)*(x[2, :] - x[0, :])
        v[N-2, :] = (sampling_rate/2)*(x[N-1, :] - x[N-3, :])
    else:
        if not sample_diff:
            v[1:N-1,] = (sampling_rate/2)*(x[2:N, :] - x[0:N-2, :])
        else:
            v[1:N,] = (sampling_rate)*(x[1:N, :] - x[0:N-1, :])
    return v
