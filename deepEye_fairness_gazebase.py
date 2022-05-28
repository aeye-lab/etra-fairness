import argparse
import os
import random
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from Evaluation import evaluation
from Preprocessing import preprocessing


def boolean_string(s: str) -> bool:
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def list_string(s: str) -> List[Any]:
    split_string = s.split(',')
    out_list = []
    for split_elem in split_string:
        out_list.append(
            split_elem.strip().replace(
                '\'', '',
            ).replace('[', '').replace(']', ''),
        )
    return out_list


def list_int(s: str) -> List[int]:
    split_string = s.split(',')
    out_list = []
    for split_elem in split_string:
        out_list.append(
            int(
                split_elem.strip().replace(
                    '\'', '',
                ).replace('[', '').replace(']', ''),
            ),
        )
    return out_list


def main() -> int:
    random_state = 42
    flag_train_on_gpu = True
    batch_size = 64
    Y_columns = {
        'subId': 0,
        'session': 1,
        'round': 2,
        'trial': 3,
    }

    # XXX: change path
    demo_info_df = pd.read_excel(
        'GazeBase_v2_0/GazeBaseDemoInfo.xlsx',
    )

    demo_list = [
        'Age',
        'Self-Identified Gender',
        'Self-Identified Ethnicity',
    ]
    demo_dict: Dict[str, Dict[int, int]] = dict()
    part_age_dict = dict()
    part_gender_dict = dict()
    part_ethnicity_dict = dict()
    for i in range(len(demo_info_df)):
        cur_line = demo_info_df.iloc[i]
        cur_part = cur_line['Participant ID']
        cur_age = cur_line['Age']
        cur_gender = cur_line['Self-Identified Gender']
        cur_ethnicity = cur_line['Self-Identified Ethnicity']
        for demo_type in demo_list:
            if demo_type not in demo_dict:
                demo_dict[demo_type] = dict()
            demo_dict[demo_type][cur_part] = cur_line[demo_type]
        part_age_dict[cur_part] = cur_age
        part_gender_dict[cur_part] = cur_gender
        part_ethnicity_dict[cur_part] = cur_ethnicity

    parser = argparse.ArgumentParser()
    # random, if we want to randomly sample the train persons
    parser.add_argument(
        '-inspect_key', '--inspect_key',
        type=str, default='Self-Identified Gender',
        choices=[
            'Self-Identified Gender' 'Age',
            'Self-Identified Ethnicity', 'random',
        ],
        help='Choose which setting to investigate  %(default)s',
    )
    parser.add_argument(
        '-inspect_list', '--inspect_list',
        default="['Male','Female']",
        help='Inspection list for the corresponding inspection key %(default)s',
    )
    # 1 = (0,1.1,0.1); 2 = [0.5]; else -1
    parser.add_argument(
        '-use_percentages',
        '--use_percentages', type=int, default=1,
        help='Number of percerntages of specific demographic to use',
    )
    parser.add_argument(
        '-use_trial_types',
        '--use_trial_types', type=str, default="['TEX']",
        help='Stimulus of the recording of the eye tracking data',
        choices=['TEX', 'RAN', 'BLG', 'VD1', 'VD2', 'HSS', 'FXS'],
    )
    parser.add_argument(
        '-number_train', '--number_train',
        type=int, default=100,
        help='Specify number of training subjects',
    )
    parser.add_argument(
        '-seconds_per_user',
        '--seconds_per_user', type=int, default=80,
        help='Time duration of each subject used during training',
    )
    parser.add_argument('-num_folds', '--num_folds', type=int, default=10)
    parser.add_argument(
        '-save_dir', '--save_dir',
        type=str, default='saved_embeddings/',
    )
    parser.add_argument('-GPU', '--GPU', type=int, default=1)
    parser.add_argument(
        '-max_round', '--max_round', type=int, default=4,
        help='Maximum amount of experiment rounds used during evaluation',
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    parser.add_argument(
        '-inspect_threshold',
        '--inspect_threshold', type=int, default=20,
        help="Age threshold to differentiate between 'young' and 'old'",
    )
    args = parser.parse_args()

    inspect_key = args.inspect_key
    inspect_list = list_string(args.inspect_list)
    use_percentages = args.use_percentages
    if use_percentages == 1:
        percentages = np.arange(0, 1.1, 0.1)
    elif use_percentages == 2:
        percentages = [0.5]
    else:
        percentages = -1
    use_trial_types = list_string(args.use_trial_types)
    number_train = args.number_train
    seconds_per_user = args.seconds_per_user
    num_folds = args.num_folds
    save_dir = args.save_dir
    GPU = args.GPU
    max_round = args.max_round
    inspect_threshold = args.inspect_threshold

    if flag_train_on_gpu:
        import tensorflow as tf
        # select graphic card
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        config = tf.compat.v1.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        tf_session = tf.compat.v1.Session(config=config)  # noqa: F841

    # XXX: change path
    gaze_base_raw_data_dirs = [
        'GazeBase_v2_0/Round_9/',
        'GazeBase_v2_0/Round_8/',
        'GazeBase_v2_0/Round_7/',
        'GazeBase_v2_0/Round_6/',
        'GazeBase_v2_0/Round_5/',
        'GazeBase_v2_0/Round_4/',
        'GazeBase_v2_0/Round_3/',
        'GazeBase_v2_0/Round_2/',
        'GazeBase_v2_0/Round_1/',
    ]
    csv_files = []
    for cur_dir in gaze_base_raw_data_dirs:
        csv_files += preprocessing.get_csvs(cur_dir)
    print('number of csv files. ' + str(len(csv_files)))

    round_list = []
    subject_list = []
    session_list = []
    trial_list = []
    path_list = []
    use_for_train = []
    for csv_file in csv_files:
        file_name = csv_file.split('/')[-1]
        file_name_split = file_name.replace('.csv', '').split('_')
        cur_round = file_name_split[1][0]
        cur_subject = int(file_name_split[1][1:])
        cur_session = file_name_split[2]
        cur_trial = file_name_split[3]
        if cur_trial not in use_trial_types:
            continue
        if int(cur_round) > max_round:
            continue
        if inspect_key != 'Age' and inspect_key != 'random':
            if demo_dict[inspect_key][cur_subject] not in inspect_list:
                use_for_train.append(0)
            else:
                use_for_train.append(1)
        else:
            use_for_train.append(1)
        round_list.append(cur_round)
        subject_list.append(cur_subject)
        session_list.append(cur_session)
        trial_list.append(cur_trial)
        path_list.append(csv_file)

    data_csv = pd.DataFrame({
        'round': round_list,
        'subject': subject_list,
        'session': session_list,
        'trial': trial_list,
        'path': path_list,
    })
    data_csv.head()

    user_data_list = []
    for i in tqdm(range(len(data_csv))):
        cur_line = data_csv.iloc[i]
        cur_path = cur_line['path']
        try:
            user_data_list.append(preprocessing.get_data_for_user(cur_path))
        except ValueError:
            print('error with file: ' + str(cur_path))

    round_list = list(np.array(data_csv['round'], dtype=np.int32))
    subject_list = list(np.array(data_csv['subject'], dtype=np.int32))
    session_list = list(np.array(data_csv['session']))
    trial_list = list(np.array(data_csv['trial']))

    number_add = 20000
    participants_per_session_dict: Dict[int, List[int]] = dict()
    X = np.zeros([number_add, 1000, 2])
    Y = np.zeros([number_add, 4])
    key_label = []
    session_nr_dict: Dict[str, int] = dict()
    trial_nr_dict: Dict[str, int] = dict()
    counter = 0
    key_sub_label = []
    use_train_ids = []
    for i in tqdm(range(len(user_data_list))):
        cur_id = i
        cur_subject = subject_list[cur_id]
        cur_session = session_list[cur_id]
        if inspect_key != 'random':
            cur_key_label = demo_dict[inspect_key][cur_subject]
        else:
            cur_key_label = 1
        if cur_session not in session_nr_dict:
            session_nr_dict[cur_session] = len(session_nr_dict)
        cur_round = round_list[cur_id]
        cur_trial = trial_list[cur_id]
        if cur_trial not in trial_nr_dict:
            trial_nr_dict[cur_trial] = len(trial_nr_dict)
        cur_data = user_data_list[cur_id]['X_vel_transformed']
        end_counter = counter + cur_data.shape[0]
        while X.shape[0] <= end_counter:
            X = np.vstack(
                [X, np.zeros([number_add, cur_data.shape[1], cur_data.shape[2]])],
            )
            Y = np.vstack([Y, np.zeros([number_add, 4])])
        X[counter:end_counter] = cur_data
        Y[counter:end_counter, 0] = cur_subject
        Y[counter:end_counter, 1] = session_nr_dict[cur_session]
        Y[counter:end_counter, 2] = cur_round
        Y[counter:end_counter, 3] = trial_nr_dict[cur_trial]
        key_label += list(cur_key_label for a in range(cur_data.shape[0]))
        counter += cur_data.shape[0]
        if session_nr_dict[cur_session] not in participants_per_session_dict:
            participants_per_session_dict[session_nr_dict[cur_session]] = []
        participants_per_session_dict[session_nr_dict[cur_session]].append(
            cur_subject,
        )
        key_sub_label.append(cur_key_label)
        use_train_label = use_for_train[cur_id]
        use_train_ids += list(
            use_train_label for a in range(cur_data.shape[0])
        )
    X = X[0:counter]
    Y = Y[0:counter]

    print(trial_nr_dict)

    for key in participants_per_session_dict.keys():
        participants_per_session_dict[key] = list(
            set(participants_per_session_dict[key]),
        )
    key_label = np.array(key_label)
    key_sub_label = np.array(key_sub_label)
    use_train_ids = np.array(use_train_ids)

    # restrict to 'seconds_per_user' per user
    random.seed(random_state)

    unique_user_list = list(np.unique(Y[:, Y_columns['subId']]))
    use_ids = []
    for unique_user in unique_user_list:
        cur_user_ids = list(
            np.where(Y[:, Y_columns['subId']] == unique_user)[0],
        )
        random.shuffle(cur_user_ids)
        cur_user_ids = cur_user_ids[0:seconds_per_user]
        use_ids += cur_user_ids

    X = X[use_ids]
    Y = Y[use_ids]
    key_label = key_label[use_ids]  # type: ignore

    unique_user_list = np.array(list(np.unique(Y[:, Y_columns['subId']])))
    key_label_list = []
    for user in unique_user_list:
        if inspect_key != 'random':
            key_label_list.append(demo_dict[inspect_key][user])
        else:
            key_label_list.append(1)
    key_label_list = np.array(key_label_list)

    if inspect_key == 'random':
        use_percentages = [1]
    else:
        use_percentages = []
        if type(percentages) == int and percentages == -1:
            if inspect_key == 'Age':
                use_percentages.append(
                    [np.round(1/2, decimals=1) for a in range(2)],
                )
            else:
                use_percentages.append(
                    [
                        np.round(1/len(inspect_list), decimals=1)
                        for a in range(len(inspect_list))
                    ],
                )
        else:
            if inspect_key == 'Age':
                for percentage in percentages:
                    use_percentages.append([
                        np.round(percentage, decimals=1),
                        np.round(1-percentage, decimals=1),
                    ])
            else:
                for percentage in percentages:
                    use_percentages.append([
                        np.round(percentage, decimals=1),
                        np.round(1-percentage, decimals=1),
                    ])

    for fold_nr in range(num_folds):
        for percentage in use_percentages:
            if inspect_key != 'Age' and inspect_key != 'random':
                save_path = save_dir +\
                    'key_' + inspect_key.replace(' ', '_') +\
                    '_list_' + str(inspect_list).replace(' ', '_') +\
                    '_trials_' + str(use_trial_types).replace(' ', '_') +\
                    '_fold_' + str(fold_nr) +\
                    '_percentage_' + str(percentage) +\
                    '_seconds_per_user_' + str(seconds_per_user) +\
                    '.npz'
            elif inspect_key == 'Age':
                save_path = save_dir +\
                    'key_' + inspect_key.replace(' ', '_') +\
                    '_threshold_' + str(inspect_threshold).replace(' ', '_') +\
                    '_trials_' + str(use_trial_types).replace(' ', '_') +\
                    '_fold_' + str(fold_nr) +\
                    '_percentage_' + str(percentage) +\
                    '_seconds_per_user_' + str(seconds_per_user) +\
                    '.npz'
            elif inspect_key == 'random':
                save_path = save_dir +\
                    'key_' + inspect_key.replace(' ', '_') +\
                    '_trials_' + str(use_trial_types).replace(' ', '_') +\
                    '_fold_' + str(fold_nr) +\
                    '_percentage_' + str(percentage) +\
                    '_seconds_per_user_' + str(seconds_per_user) +\
                    '.npz'
            random.seed(fold_nr)
            random_ids = np.arange(len(unique_user_list))
            random.shuffle(random_ids)
            unique_user_list = unique_user_list[random_ids]
            key_label_list = key_label_list[random_ids]

            # dictionary to store for each key of inspect_list the
            # users belonging to specific key from inspect_list
            key_user_dict: Dict[int, List[int]] = dict()
            for i in range(len(unique_user_list)):
                cur_user = unique_user_list[i]
                cur_key = key_label_list[i]
                if inspect_key == 'Age':
                    cur_key = int(cur_key <= inspect_threshold)
                if cur_key not in key_user_dict:
                    key_user_dict[cur_key] = []
                key_user_dict[cur_key].append(cur_user)

            '''
            for key_user in key_user_dict:
                print(str(key_user) + ': ' + str(len(key_user_dict[key_user])))
            '''

            train_user = []
            test_user = []
            if inspect_key == 'Age':
                counter = 0
                for key in key_user_dict:
                    train_user_end = int(
                        np.round(percentage[counter] * number_train),
                    )
                    train_user += list(key_user_dict[key][0:train_user_end])
                    counter += 1
            elif inspect_key == 'random':
                train_user = unique_user_list[0:number_train]
            else:
                for i in range(len(inspect_list)):
                    key = inspect_list[i]
                    train_user_end = int(
                        np.round(percentage[i] * number_train),
                    )
                    train_user += list(key_user_dict[key][0:train_user_end])

            test_ids = ~ np.isin(Y[:, Y_columns['subId']], train_user)
            train_ids = np.isin(Y[:, Y_columns['subId']], train_user)
            test_user = list(np.unique(Y[test_ids, Y_columns['subId']]))

            print('number of train user: ' + str(len(train_user)))
            print('number of test user: ' + str(len(test_user)))

            X_train = X[train_ids]
            Y_train = Y[train_ids]
            X_test = X[test_ids]
            Y_test = Y[test_ids]

            print('keys for train: ' + str(np.unique(key_label[train_ids])))
            print('keys for test ' + str(np.unique(key_label[test_ids])))
            # double the input for left and right eye
            X_train = np.concatenate([X_train, X_train], axis=2)
            X_test = np.concatenate([X_test, X_test], axis=2)
            embeddings, model = evaluation.evaluate_create_test_embeddings(
                X_train,
                Y_train,
                X_test, Y_test,
                Y_columns,
                batch_size=batch_size,
                return_model=True,
            )
            np.savez_compressed(
                save_path,
                embeddings=embeddings,
                Y_test=Y_test,
                train_user=train_user,
                test_user=test_user,
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
