import argparse
import os
from typing import List

import joblib
import numpy as np
from tqdm import tqdm

from Evaluation import evaluation


Y_columns = {
    'subId': 0,
    'session': 1,
    'round': 2,
    'trial': 3,
}

Y_columns_judo = {
    'subId': 0,
    'session': 1,
    'trialId': 2,
    'seqId': 3,
    'original_trial_length_before_padding': 4,
}


def create_score_dicts_lohr(
    npz_embedding_dir: np.array, score_dict_save_dir: str, flag_redo: bool = False,
    window_sizes: List[str] = ['all'],
    random_state: int = 42,
) -> None:
    convert_in_files = []
    convert_out_files = []
    list_dir = os.listdir(npz_embedding_dir)
    for cur_path in list_dir:
        if cur_path.endswith('.npz'):
            cur_complete_path = npz_embedding_dir + '/' + cur_path
            cur_complete_save_path = score_dict_save_dir + \
                '/' + cur_path.replace('.npz', '.joblib')
            if not flag_redo:
                if os.path.exists(cur_complete_save_path):
                    continue
            convert_in_files.append(cur_complete_path)
            convert_out_files.append(cur_complete_save_path)

    for i in tqdm(np.arange(len(convert_in_files))):
        # get all embeddings
        cur_data = np.load(convert_in_files[i])
        embeddings = cur_data['embeddings']
        Y_test = cur_data['Y_test']

        ########################################
        #
        # CREATE SCORES FOR GAZEBASE
        #
        ########################################

        score_dicts, label_dicts, person_one_dicts, person_two_dicts = evaluation.get_scores_and_labels_from_raw_lohr(  # noqa: E501
            embeddings=embeddings,
            Y_test=Y_test,
            Y_columns=Y_columns,
            window_sizes=window_sizes,
            random_state=random_state,
            session_key='session',
            seq_id_key='round',
            subject_key='subId',
            verbose=0,
        )

        joblib.dump(
            {
                'score_dicts': score_dicts,
                'label_dicts': label_dicts,
                'person_one_dicts': person_one_dicts,
                'person_two_dicts': person_two_dicts,
            }, convert_out_files[i], compress=3, protocol=2,
        )


#    0.77 sac + 0.08 fix + 0.15 pso
def create_score_dicts_lohr_merged(
    npz_embedding_dir: np.array, score_dict_save_dir: str,
    flag_redo: bool = False,
    window_sizes: List[str] = ['all'],
    random_state: int = 42,
    weight_list: List[float] = [.77, .08, .15],
    feature_list: List[str] = ['sac', 'fix', 'pso'],
) -> None:
    convert_in_files = []
    convert_out_files = []
    list_dir = os.listdir(npz_embedding_dir)
    for cur_path in list_dir:
        if cur_path.endswith('.npz'):
            cur_complete_path = npz_embedding_dir + '/' + cur_path
            if feature_list[0] in cur_complete_path:
                cur_convert_list = []
                for j in range(len(feature_list)):
                    tmp_path = cur_complete_path.replace(
                        feature_list[0], feature_list[j],
                    )
                    if os.path.exists(tmp_path):
                        cur_convert_list.append(tmp_path)
                if len(cur_convert_list) == len(feature_list):
                    convert_save_path = score_dict_save_dir + '/' + \
                        cur_path.replace('.npz', '.joblib').replace(
                            feature_list[0], 'merged',
                        )
                    if not flag_redo:
                        if os.path.exists(convert_save_path):
                            continue
                    convert_in_files.append(cur_convert_list)
                    convert_out_files.append(convert_save_path)

    for i in tqdm(np.arange(len(convert_in_files))):
        # get all embeddings
        embedding_list = []
        Y_test_list = []
        cur_convert_files = convert_in_files[i]
        for j in range(len(cur_convert_files)):
            cur_data = np.load(cur_convert_files[j])
            embeddings = cur_data['embeddings']
            Y_test = cur_data['Y_test']
            embedding_list.append(embeddings)
            Y_test_list.append(Y_test)
            (score_dicts, label_dicts, person_one_dicts, person_two_dicts) = evaluation.get_scores_and_labels_from_raw_lohr_merged(  # noqa: E501
                embedding_list,
                Y_test_list,
                weight_list,
                Y_columns,
                window_sizes=['all'],
                random_state=None,
                session_key='session',
                seq_id_key='round',
                subject_key='subId',
                verbose=0,
            )
        joblib.dump(
            {
                'score_dicts': score_dicts,
                'label_dicts': label_dicts,
                'person_one_dicts': person_one_dicts,
                'person_two_dicts': person_two_dicts,
            }, convert_out_files[i], compress=3, protocol=2,
        )


def create_score_dicts(
    npz_embedding_dir: np.array, score_dict_save_dir: str, flag_redo: bool = False,
    window_sizes: List[int] = [10], random_state: int = 42,
) -> None:
    convert_in_files = []
    convert_out_files = []
    list_dir = os.listdir(npz_embedding_dir)
    for cur_path in list_dir:
        if cur_path.endswith('.npz'):
            cur_complete_path = npz_embedding_dir + '/' + cur_path
            cur_complete_save_path = score_dict_save_dir + \
                '/' + cur_path.replace('.npz', '.joblib')
            if not flag_redo:
                if os.path.exists(cur_complete_save_path):
                    continue
            convert_in_files.append(cur_complete_path)
            convert_out_files.append(cur_complete_save_path)

    for i in tqdm(np.arange(len(convert_in_files))):
        # get all embeddings
        cur_data = np.load(convert_in_files[i])
        embeddings = cur_data['embeddings']
        Y_test = cur_data['Y_test']
        test_user = cur_data['test_user']

        ########################################
        #
        # CREATE SCORES FOR GAZEBASE
        #
        ########################################

        test_subs = list(np.unique(Y_test[:, Y_columns['subId']]))

        # setup for the calculation of scores
        n_train_users = 0
        n_enrolled_users = len(test_subs)
        n_impostors = 0
        n_enrollment_sessions = 1
        n_test_sessions = 1
        test_user = None
        test_sessions = None
        user_test_sessions = None
        enrollment_sessions = None
        verbose = 0
        seconds_per_session = None
        session_key = 'session'
        seq_id_key = 'round'

        (score_dicts, label_dicts, person_one_dicts, person_two_dicts) = evaluation.get_scores_and_labels_from_raw(  # noqa: E501
            test_embeddings=embeddings,
            Y_test=Y_test,
            Y_columns=Y_columns,
            window_sizes=window_sizes,
            n_train_users=n_train_users,
            n_enrolled_users=n_enrolled_users,
            n_impostors=n_impostors,
            n_enrollment_sessions=n_enrollment_sessions,
            n_test_sessions=n_test_sessions,
            test_user=test_user,
            test_sessions=test_sessions,
            user_test_sessions=user_test_sessions,
            enrollment_sessions=enrollment_sessions,
            verbose=verbose,
            random_state=random_state,
            seconds_per_session=seconds_per_session,
            session_key=session_key,
            seq_id_key=seq_id_key,
        )

        joblib.dump(
            {
                'score_dicts': score_dicts,
                'label_dicts': label_dicts,
                'person_one_dicts': person_one_dicts,
                'person_two_dicts': person_two_dicts,
            }, convert_out_files[i], compress=3, protocol=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-npz_embedding_dir', '--npz_embedding_dir',
        type=str, default='saved_embeddings/',
    )
    parser.add_argument(
        '-score_dict_save_dir', '--score_dict_save_dir',
        type=str, default='saved_score_dicts/',
    )

    args = parser.parse_args()
    npz_embedding_dir = args.npz_embedding_dir
    score_dict_save_dir = args.score_dict_save_dir
    create_score_dicts(npz_embedding_dir, score_dict_save_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
