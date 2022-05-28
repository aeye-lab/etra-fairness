import argparse
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from Evaluation import evaluation
from Preprocessing import preprocessing

# 0.77 sac + 0.08 fix + 0.15 pso


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def list_string(s):
    split_string = s.split(',')
    out_list = []
    for split_elem in split_string:
        out_list.append(
            split_elem.strip().replace(
                '\'', '',
            ).replace('[', '').replace(']', ''),
        )
    return out_list


def list_int(s):
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


def main():
    inspect_key = 'random'
    num_folds = 10
    flag_train_on_gpu = True
    step_size = 200000
    flag_use_min_max = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-use_trial_types',
        '--use_trial_types', type=str, default="['TEX']",
    )
    parser.add_argument(
        '-number_train', '--number_train',
        type=int, default=100,
    )  # 50
    parser.add_argument('-num_folds', '--num_folds', type=int, default=10)
    parser.add_argument(
        '-save_dir', '--save_dir', type=str,
        default='saved_lohr_embeddings/',
    )
    parser.add_argument('-GPU', '--GPU', type=int, default=1)
    parser.add_argument('-max_round', '--max_round', type=int, default=4)
    parser.add_argument('-batchsize', '--batchsize', type=int, default=100)
    parser.add_argument('-save_prefix', '--save_prefix', type=str, default='')
    parser.add_argument(
        '-optimizer', '--optimizer', type=str,
        default='adam_w',
    )   # adam_w or adam
    parser.add_argument(
        '-feature', '--feature', type=str,
        default='Fix',
    )   # Fix or Sac or PSO
    # XXX: change path
    parser.add_argument(
        '-demo_path', '--demo_path', type=str,
        default='GazeBase_v2_0/GazeBaseDemoInfo.xlsx',
    )
    parser.add_argument(
        '-use_data_generator',
        '--use_data_generator', type=int, default=1,
    )
    parser.add_argument(
        '-use_percentage',
        '--use_percentage', type=float, default=-1.,
    )
    parser.add_argument(
        '-inspect_key', '--inspect_key',
        type=str, default='random',
    )  # Self-Identified Ethnicity'
    parser.add_argument(
        '-inspect_list', '--inspect_list',
        type=str, default="['White','Hispanic']",
    )

    args = parser.parse_args()
    use_trial_types = list_string(args.use_trial_types)
    number_train = args.number_train
    num_folds = args.num_folds
    save_dir = args.save_dir
    max_round = args.max_round
    save_prefix = args.save_prefix
    optimizer = args.optimizer
    feature = args.feature
    GPU = args.GPU
    demo_path = args.demo_path
    batch_size = args.batchsize
    use_data_generator = args.use_data_generator
    if use_data_generator == 1:
        use_data_generator = True
    else:
        use_data_generator = False
    use_percentage = args.use_percentage
    inspect_list = list_string(args.inspect_list)
    inspect_key = args.inspect_key

    epochs = int(np.round(step_size / batch_size))

    demo_info_df = pd.read_excel(demo_path)
    unique_user_list = np.array(
        list(np.unique(demo_info_df['Participant ID'])), dtype=np.int32,
    )

    demo_list = [
        'Age',
        'Self-Identified Gender',
        'Self-Identified Ethnicity',
    ]
    demo_dict = dict()
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

    if flag_train_on_gpu:
        import tensorflow as tf
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        config = tf.compat.v1.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        tf_session = tf.compat.v1.Session(config=config)  # noqa: F841

    # XXX: change path
    gaze_base_feature_data_dirs = [
        # only works locally see README.md for more info
        '',
    ]
    csv_files = []
    for cur_dir in gaze_base_feature_data_dirs:
        csv_files += preprocessing.get_lohr_csvs(cur_dir)
    print('number of csv files. ' + str(len(csv_files)))

    round_list = []
    subject_list = []
    session_list = []
    trial_list = []
    path_list = []
    use_for_train = []
    feature_list = []
    for csv_file in csv_files:
        file_name = csv_file.split('/')[-1]
        file_name_split = file_name.replace('.csv', '').split('_')
        cur_round = file_name_split[2][0]
        cur_subject = int(file_name_split[2][1:])
        cur_session = file_name_split[3]
        cur_trial = file_name_split[4]
        cur_feature_name = file_name_split[6]
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
        feature_list.append(cur_feature_name)

    data_csv = pd.DataFrame({
        'round': round_list,
        'subject': subject_list,
        'session': session_list,
        'trial': trial_list,
        'path': path_list,
        'feature': feature_list,
    })

    user_data_list = []
    use_ids = []
    for i in tqdm(range(len(data_csv))):
        cur_line = data_csv.iloc[i]
        cur_path = cur_line['path']
        try:
            user_data_list.append(
                preprocessing.get_data_for_user_lohr(cur_path),
            )
            use_ids.append(i)
        except FileNotFoundError:
            continue

    round_list = list(np.array(data_csv['round'], dtype=np.int32)[use_ids])
    subject_list = list(np.array(data_csv['subject'], dtype=np.int32)[use_ids])
    session_list = list(np.array(data_csv['session'])[use_ids])
    trial_list = list(np.array(data_csv['trial'])[use_ids])
    feature_list = list(np.array(data_csv['feature'])[use_ids])

    feature_input_size_dict = {
        'Fix': 61,
        'Sac': 81,
        'PSO': 44,
    }

    Y_columns = {
        'subId': 0,
        'session': 1,
        'round': 2,
        'trial': 3,
    }

    number_add = 2000000
    X = np.zeros([number_add, feature_input_size_dict[feature]])
    Y = np.zeros([number_add, 5])
    session_nr_dict = dict()
    trial_nr_dict = dict()
    counter = 0
    for i in tqdm(range(len(user_data_list))):
        cur_id = i
        cur_subject = subject_list[cur_id]
        cur_session = session_list[cur_id]
        cur_feature = feature_list[cur_id]
        if cur_feature != feature:
            continue
        if cur_session not in session_nr_dict:
            session_nr_dict[cur_session] = len(session_nr_dict)
        cur_round = round_list[cur_id]
        cur_trial = trial_list[cur_id]
        if cur_trial not in trial_nr_dict:
            trial_nr_dict[cur_trial] = len(trial_nr_dict)
        cur_data = user_data_list[cur_id]['X']
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
        counter += cur_data.shape[0]
    X = X[0:counter]
    Y = Y[0:counter]
    X = np.nan_to_num(X, nan=0)

    unique_user, number_user = np.unique(
        Y[:, Y_columns['subId']], return_counts=True,
    )

    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    unique_user_list = np.array(list(np.unique(Y[:, Y_columns['subId']])))
    key_label_list = []
    for user in unique_user_list:
        if inspect_key != 'random':
            key_label_list.append(demo_dict[inspect_key][user])
        else:
            key_label_list.append(1)
    key_label_list = np.array(key_label_list)

    for fold_nr in range(num_folds):
        if use_percentage == -1.:
            save_path = save_dir +\
                save_prefix +\
                'key_' + inspect_key.replace(' ', '_') +\
                '_trials_' + str(use_trial_types).replace(' ', '_') +\
                '_fold_' + str(fold_nr) +\
                '.npz'
        else:
            save_path = save_dir +\
                save_prefix +\
                'key_' + inspect_key.replace(' ', '_') +\
                '_trials_' + str(use_trial_types).replace(' ', '_') +\
                '_fold_' + str(fold_nr) +\
                '_percentage_' + str(use_percentage) +\
                '.npz'

        if use_percentage == -1:
            random.seed(fold_nr)
            random_ids = np.arange(len(unique_user))
            random.shuffle(random_ids)
            unique_user = unique_user[random_ids]

            train_user = []
            test_user = []
            train_user = unique_user[0:number_train]

            test_ids = ~ np.isin(Y[:, Y_columns['subId']], train_user)
            train_ids = np.isin(Y[:, Y_columns['subId']], train_user)
            test_user = list(np.unique(Y[test_ids, Y_columns['subId']]))

            print('number of train user: ' + str(len(train_user)))
            print('number of test user: ' + str(len(test_user)))

        else:
            random.seed(fold_nr)
            random_ids = np.arange(len(unique_user_list))
            random.shuffle(random_ids)
            unique_user_list = unique_user_list[random_ids]
            key_label_list = key_label_list[random_ids]

            # dictionary to store for each key of inspect_list the
            # users belonging to specific key from inspect_list
            key_user_dict = dict()
            for i in range(len(unique_user_list)):
                cur_user = unique_user_list[i]
                cur_key = key_label_list[i]
                if cur_key not in key_user_dict:
                    key_user_dict[cur_key] = []
                key_user_dict[cur_key].append(cur_user)

            train_user = []
            test_user = []
            percentage = [use_percentage, 1. - use_percentage]
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

        if flag_use_min_max:
            X_train = X_scaled[train_ids]
            Y_train = Y[train_ids]
            X_test = X_scaled[test_ids]
            Y_test = Y[test_ids]
        else:
            X_train = X[train_ids]
            Y_train = Y[train_ids]
            X_test = X[test_ids]
            Y_test = Y[test_ids]

        print('X_train.shape: ' + str(X_train.shape))
        print('X_test.shape: ' + str(X_test.shape))

        # shuffle data
        rand_ids = np.arange(X_train.shape[0])
        random.shuffle(rand_ids)
        X_train = X_train[rand_ids]
        Y_train = Y_train[rand_ids]

        # double the input for left and right eye
        embeddings, model_lohr = evaluation.evaluate_create_test_embeddings(
            X_train, Y_train,
            X_test, Y_test,
            Y_columns,
            batch_size=batch_size,
            return_model=True,
            model='lohr',
            feature=feature,
            optimizer=optimizer,
            use_data_generator=use_data_generator,
            epochs=epochs,
        )

        np.savez_compressed(
            save_path,
            embeddings=embeddings,
            Y_test=Y_test,
            train_user=train_user,
            test_user=test_user,
        )


if __name__ == '__main__':
    raise SystemExit(main())
