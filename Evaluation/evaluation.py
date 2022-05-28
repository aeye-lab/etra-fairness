import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from Model import configuration
from Model import deepeyedentification_tf2
from Model import lohr_model


model_conf = {
    'learning_rate_merged': 0.00011, 'kernel_sub2': [9, 9, 9, 5, 5, 5, 5, 3, 3],
    'normalization_sub2': ['zscore'], 'kernel_sub1': [9, 9, 9, 5, 5, 5, 5, 3, 3],
    'transform_sub2': ['clip', 0.01], 'filters_sub1': [128, 128, 128, 256, 256, 256, 256, 256, 256],
    'dense_sub1': [256, 256, 128], 'learning_rate_sub2': 0.001, 'normalization_sub1': 'None',
    'name_sub1': 'optimal_slow_subnet', 'learning_rate_sub1': 0.001,
    'filters_sub2': [32, 32, 32, 512, 512, 512, 512, 512, 512], 'name_merged': 'optimal_merged',
    'Ndense_merged': [256, 128], 'dense_sub2': [256, 256, 128],
    'strides_sub1': [1, 1, 1, 1, 1, 1, 1, 1, 1], 'transform_sub1': ['tanh', 20.0],
    'name_sub2': 'optimal_fast_subnet', 'strides_sub2': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    'batch_size': 64,
}


FEATURE_INPUT_SIZE_DICT = {
    'Fix': 61,
    'Sac': 81,
    'PSO': 44,
}


def get_indicies(
    enrolled_users: np.array,
    impostors: np.array,
    enrollment_sessions: np.array,
    test_sessions: np.array,
    data_user: np.array,
    data_sessions: np.array,
    data_seqIds: np.array,
    seconds_per_session: Optional[float] = None,
    random_state: Optional[int] = 42,
    num_enrollment: int = 12,
) -> Tuple[List[int], np.array]:

    random.seed(random_state)
    idx_enrollment = []
    for enrolled_user in enrolled_users:
        cur_ids = np.logical_and(
            np.isin(data_user, enrolled_user),
            np.isin(data_sessions, enrollment_sessions),
        )
        pos_ids = np.where(cur_ids)[0]
        random.shuffle(pos_ids)
        idx_enrollment += list(pos_ids[0:num_enrollment])

    test_idx = np.logical_and(
        np.logical_or(
            np.isin(data_user, enrolled_users),
            np.isin(data_user, impostors),
        ),
        np.isin(data_sessions, test_sessions),
    )

    return (idx_enrollment, test_idx)


def get_user_similarity_scores_and_labels(
    cosine_distances: np.array, y_enrollment: np.array, y_test: np.array,
    enrollment_users: np.array, impostors: np.array, window_size: Union[int, str] = 1,
    sim_to_enroll: str = 'min', verbose: int = 0,
) -> Tuple[np.array, ...]:
    """

    :param cosine_distances:
        cosine distances of all pairs of enrollment and test instances, n_test x n_enrollment
    :param y_enrollment: n_enrollment labels for enrollment instances
    :param y_test: n_test labels for test instances
    :param enrollment_users: all ids of enrolled users
    :param impostors: all ids of impostors
    :param window_size: number of instances the similarity score should be based upon
    :param sim_to_enroll: how to compute simalarity to enrollment users; should be in {'min','mean'}
    :return:
        similarity scores of two persons;
        true labels: test person is impostor (0), same person (1) or another enrolled person (2)
    """
    if verbose == 0:
        disable = True
    else:
        disable = False

    # similarity score between two users, based on number of test instances specified by window size
    scores = []
    # true labels: test person is 0 (impostor),1 (correct), 2 (confused)
    labels = []
    person_one = []
    person_two = []
    for test_user in tqdm(np.unique(y_test), disable=disable):
        idx_test_user = y_test == test_user

        # iterate over each possible window start position for test user
        dists_test_user = cosine_distances[idx_test_user, :]
        if str(window_size) != 'all':
            for i in range(dists_test_user.shape[0] - window_size):
                dists_test_user_window = dists_test_user[i:i+window_size, :]

                # calculate score and prediction and create true label for each window
                distances_to_enrolled = []
                enrolled_u = []
                enrolled_persons = np.unique(y_enrollment)

                for enrolled_user in enrolled_persons:
                    idx_enrolled_user = y_enrollment == enrolled_user

                    # calculate aggregated distance of instances in window with each enrolled user
                    dists_test_user_window_enrolled_user = dists_test_user_window[
                        :, idx_enrolled_user
                    ]

                    # aggregate distances for each test sequence to all enrolled sequences
                    # by taking the minimum distance
                    if sim_to_enroll == 'min':
                        dists_test_sequences_of_window = np.min(
                            dists_test_user_window_enrolled_user, axis=1,
                        )  # n_test_sequences x 1 array
                    elif sim_to_enroll == 'mean':
                        dists_test_sequences_of_window = np.mean(
                            dists_test_user_window_enrolled_user, axis=1,
                        )  # n_test_sequences x 1 array

                    # aggregate min distances of all test sequences in the window by taking the mean
                    window_mean_dist = np.mean(dists_test_sequences_of_window)

                    distances_to_enrolled.append(window_mean_dist)
                    enrolled_u.append(enrolled_user)

                    # create corresponding true label for this window
                    if test_user in list(impostors):
                        label = 0  # test user of this window is an impostor
                    elif test_user in list(enrollment_users):
                        if test_user == enrolled_user:
                            label = 1  # test user of this window is this enrolled user
                        else:
                            label = 2  # test user of this window is another enrolled user
                    else:
                        print(
                            f'user {test_user} is neither enrolled user nor impostor',
                        )
                        label = -1  # should never happen

                    scores.append(1-window_mean_dist)
                    labels.append(label)
                    person_one.append(enrolled_user)
                    person_two.append(test_user)
        else:
            dists_test_user_window = dists_test_user

            # calculate score and prediction and create true label for each window
            distances_to_enrolled = []
            enrolled_u = []
            enrolled_persons = np.unique(y_enrollment)

            for enrolled_user in enrolled_persons:
                idx_enrolled_user = y_enrollment == enrolled_user

                # calculate aggregated distance of instances in window with each enrolled user
                dists_test_user_window_enrolled_user = dists_test_user_window[
                    :, idx_enrolled_user
                ]

                # aggregate distances for each test sequence to all enrolled sequences
                # by taking the minimum distance
                if sim_to_enroll == 'min':
                    dists_test_sequences_of_window = np.min(
                        dists_test_user_window_enrolled_user, axis=1,
                    )  # n_test_sequences x 1 array
                elif sim_to_enroll == 'mean':
                    dists_test_sequences_of_window = np.mean(
                        dists_test_user_window_enrolled_user, axis=1,
                    )  # n_test_sequences x 1 array

                # aggregate min distances of all test sequences in this window by taking the mean
                window_mean_dist = np.mean(dists_test_sequences_of_window)

                distances_to_enrolled.append(window_mean_dist)
                enrolled_u.append(enrolled_user)

                # create corresponding true label for this window
                if test_user in list(impostors):
                    label = 0  # test user of this window is an impostor
                elif test_user in list(enrollment_users):
                    if test_user == enrolled_user:
                        label = 1  # test user of this window is this enrolled user
                    else:
                        label = 2  # test user of this window is another enrolled user
                else:
                    print(
                        f'user {test_user} is neither enrolled user nor impostor',
                    )
                    label = -1  # should never happen

                scores.append(1-window_mean_dist)
                labels.append(label)
                person_one.append(enrolled_user)
                person_two.append(test_user)

    return np.array(scores), np.array(labels), np.array(person_one), np.array(person_two)


def get_scores_and_labels_from_raw(
    test_embeddings: np.array, Y_test: Optional[np.array], Y_columns: Dict[str, int],
    window_sizes: List[int],
    n_train_users: int = 0,
    n_enrolled_users: int = 20,
    n_impostors: int = 5,
    n_enrollment_sessions: int = 3,
    n_test_sessions: int = 1,
    test_user: Optional[np.array] = None,
    user_test_sessions: Optional[np.array] = None,
    enrollment_sessions: Optional[np.array] = None,
    test_sessions: Optional[np.array] = None,
    verbose: int = 1,
    random_state: Optional[int] = None,
    seconds_per_session: Optional[int] = None,
    session_key: str = 'round',
    seq_id_key: str = 'session',
):
    if random_state is not None:
        random.seed(random_state)

    score_dicts = dict()
    label_dicts = dict()
    person_one_dicts = dict()
    person_two_dicts = dict()

    if Y_test is None:
        test_seqIds = None
    else:
        test_user = Y_test[:, Y_columns['subId']]
        test_sessions = Y_test[:, Y_columns[session_key]]
        try:
            test_seqIds = Y_test[:, Y_columns[seq_id_key]]
        except IndexError:
            test_seqIds = None

    users = list(np.unique(test_user))

    # shuffle users
    random.shuffle(users)

    enrolled_users = users[n_train_users:n_train_users+n_enrolled_users]
    impostors = users[
        n_train_users +
        n_enrolled_users:n_train_users+n_enrolled_users+n_impostors
    ]

    sessions = np.unique(test_sessions)
    random.shuffle(sessions)
    cur_enrollment_sessions = sessions[0:n_enrollment_sessions]
    cur_test_sessions = sessions[
        n_enrollment_sessions:
        n_enrollment_sessions + n_test_sessions
    ]

    if verbose > 0:
        print(
            f'enrolled_users: {enrolled_users} enroll-sessions: {cur_enrollment_sessions} test-sessions: {cur_test_sessions}',  # noqa: E501
        )

    (idx_enrollment, test_idx) = get_indicies(
        enrolled_users,
        impostors,
        cur_enrollment_sessions,
        cur_test_sessions,
        test_user,
        test_sessions,
        test_seqIds,
        seconds_per_session=seconds_per_session,
        random_state=random_state,
    )

    test_feature_vectors = test_embeddings[test_idx, :]
    enrollment_feature_vectors = test_embeddings[idx_enrollment, :]

    # labels for embedding feature vectors:
    y_enrollment_user = test_user[idx_enrollment]
    y_test_user = test_user[test_idx]

    dists = distance.cdist(
        test_feature_vectors,
        enrollment_feature_vectors, metric='cosine',
    )

    for window_size in window_sizes:
        scores, labels, person_one, person_two = get_user_similarity_scores_and_labels(
            dists,
            y_enrollment_user,
            y_test_user,
            enrolled_users,
            impostors,
            window_size=window_size,
            verbose=verbose,
        )
        cur_key = str(window_size)
        score_dicts[cur_key] = scores.tolist()
        label_dicts[cur_key] = labels.tolist()
        person_one_dicts[cur_key] = person_one.tolist()
        person_two_dicts[cur_key] = person_two.tolist()
    return (score_dicts, label_dicts, person_one_dicts, person_two_dicts)


def get_scores_and_labels_from_raw_lohr(
        embeddings, Y_test, Y_columns,
        window_sizes=['all'],
        random_state=None,
        session_key='session',
        seq_id_key='round',
        subject_key='subId',
        verbose=1,
):

    score_dicts = dict()
    label_dicts = dict()
    person_one_dicts = dict()
    person_two_dicts = dict()

    if random_state is not None:
        random.seed(random_state)

    score_dicts = dict()
    label_dicts = dict()
    person_one_dicts = dict()
    person_two_dicts = dict()

    test_user = Y_test[:, Y_columns[subject_key]]
    test_sessions = Y_test[:, Y_columns[session_key]]
    test_seqIds = Y_test[:, Y_columns[seq_id_key]]

    unique_rounds = list(np.unique(test_seqIds))

    for c_round in unique_rounds:
        test_feature_vectors = []
        enrollment_feature_vectors = []
        y_enrollment_user = []
        y_test_user = []
        cur_ids = np.where(test_seqIds == c_round)[0]
        cur_user = test_user[cur_ids]
        cur_embeddings = embeddings[cur_ids]
        cur_sessions = test_sessions[cur_ids]
        cur_u_user = list(np.unique(cur_user))
        for c_user in cur_u_user:
            c_user_ids = np.where(cur_user == c_user)[0]
            c_user_embeddings = cur_embeddings[c_user_ids]
            c_user_sessions = cur_sessions[c_user_ids]
            # enrollment
            enroll_ids = np.where(c_user_sessions == 0)[0]
            enroll_embeddings = np.mean(c_user_embeddings[enroll_ids], axis=0)
            enrollment_feature_vectors.append(enroll_embeddings)
            y_enrollment_user.append(c_user)
            # test
            test_ids = np.where(c_user_sessions == 1)[0]
            test_embeddings = np.mean(c_user_embeddings[test_ids], axis=0)
            test_feature_vectors.append(test_embeddings)
            y_test_user.append(c_user)
        test_feature_vectors = np.array(test_feature_vectors)
        enrollment_feature_vectors = np.array(enrollment_feature_vectors)
        y_enrollment_user = np.array(y_enrollment_user)
        y_test_user = np.array(y_test_user)

        dists = distance.cdist(
            test_feature_vectors,
            enrollment_feature_vectors, metric='cosine',
        )

        for window_size in window_sizes:
            scores, labels, person_one, person_two = get_user_similarity_scores_and_labels(
                dists,
                y_enrollment_user,
                y_test_user,
                cur_u_user,
                [],
                window_size=window_size,
                verbose=verbose,
            )
            cur_key = str(window_size)
            if cur_key not in score_dicts:
                score_dicts[cur_key] = []
                label_dicts[cur_key] = []
                person_one_dicts[cur_key] = []
                person_two_dicts[cur_key] = []

            score_dicts[cur_key] += scores.tolist()
            label_dicts[cur_key] += labels.tolist()
            person_one_dicts[cur_key] += person_one.tolist()
            person_two_dicts[cur_key] += person_two.tolist()

    return (score_dicts, label_dicts, person_one_dicts, person_two_dicts)


def get_scores_and_labels_from_raw_lohr_merged(
    embedding_list: List[Any],
    Y_test_list: List[Any],
    weight_list: List[Any],
    Y_columns: Dict[str, int],
    window_sizes: List[Union[str, int]] = ['all'],
    random_state: Optional[int] = None,
    session_key: str = 'session',
    seq_id_key: str = 'round',
    subject_key: str = 'subId',
    verbose: int = 1,
):

    # returns embeddings for specific user for specific round and session
    def get_embeddings_for_user(
        user_id, round_id, session_id,
        embeddings, rounds, sessions, users,
    ):
        use_ids = np.logical_and(
            np.isin(users, user_id),
            np.logical_and(
                np.isin(rounds, round_id),
                np.isin(sessions, session_id),
            ),
        )
        return embeddings[use_ids]

    # create weighted average
    def create_weighted_average(embedding_vectors, weights):
        mean_embeddings = []
        for i in range(len(embedding_vectors)):
            mean_embeddings.append(np.mean(embedding_vectors[i], axis=0))
        mean_embedding = []
        for i in range(len(mean_embeddings)):
            if i == 0:
                mean_embedding = weights[i] * mean_embeddings[i]
            else:
                mean_embedding += weights[i] * mean_embeddings[i]
        return mean_embedding

    score_dicts = dict()
    label_dicts = dict()
    person_one_dicts = dict()
    person_two_dicts = dict()

    if random_state is not None:
        random.seed(random_state)

    score_dicts = dict()
    label_dicts = dict()
    person_one_dicts = dict()
    person_two_dicts = dict()

    test_user_list = []
    test_sessions_list = []
    test_seqIds_list = []
    for i in range(len(embedding_list)):
        test_user_list.append(Y_test_list[i][:, Y_columns[subject_key]])
        test_sessions_list.append(Y_test_list[i][:, Y_columns[session_key]])
        test_seqIds_list.append(Y_test_list[i][:, Y_columns[seq_id_key]])

    unique_rounds = list(np.unique(test_seqIds_list[0]))
    unique_users = list(np.unique(test_user_list[0]))

    for c_round in unique_rounds:
        test_feature_vectors = []
        enrollment_feature_vectors = []
        y_enrollment_user = []
        y_test_user = []
        for c_user in unique_users:
            # get embedding vectors
            embedding_vectors = []
            lens = []
            for i in range(len(embedding_list)):
                cur_embedding = get_embeddings_for_user(
                    c_user, c_round, 0,
                    embedding_list[i], test_seqIds_list[i],
                    test_sessions_list[i], test_user_list[i],
                )
                embedding_vectors.append(cur_embedding)
                lens.append(cur_embedding.shape[0])
            if np.sum(np.array(lens) == 0) == 0:
                # create weighted average
                mean_embedding = create_weighted_average(
                    embedding_vectors, weight_list,
                )
                enrollment_feature_vectors.append(mean_embedding)
                y_enrollment_user.append(c_user)

            # get test vectors
            embedding_vectors = []
            lens = []
            for i in range(len(embedding_list)):
                cur_embedding = get_embeddings_for_user(
                    c_user, c_round, 1,
                    embedding_list[i], test_seqIds_list[i],
                    test_sessions_list[i], test_user_list[i],
                )
                embedding_vectors.append(cur_embedding)
                lens.append(cur_embedding.shape[0])
            if np.sum(np.array(lens) == 0) == 0:
                # create weighted average
                mean_embedding = create_weighted_average(
                    embedding_vectors, weight_list,
                )
                test_feature_vectors.append(mean_embedding)
                y_test_user.append(c_user)
        dists = distance.cdist(
            test_feature_vectors,
            enrollment_feature_vectors, metric='cosine',
        )

        for window_size in window_sizes:
            scores, labels, person_one, person_two = get_user_similarity_scores_and_labels(
                dists,
                y_enrollment_user,
                y_test_user,
                unique_users,
                [],
                window_size=window_size,
                verbose=verbose,
            )
            cur_key = str(window_size)
            if cur_key not in score_dicts:
                score_dicts[cur_key] = []
                label_dicts[cur_key] = []
                person_one_dicts[cur_key] = []
                person_two_dicts[cur_key] = []

            score_dicts[cur_key] += scores.tolist()
            label_dicts[cur_key] += labels.tolist()
            person_one_dicts[cur_key] += person_one.tolist()
            person_two_dicts[cur_key] += person_two.tolist()

    return (score_dicts, label_dicts, person_one_dicts, person_two_dicts)


def evaluate_create_test_embeddings(
    X_train: np.array,
    Y_train: np.array,
    X_test: np.array,
    Y_test: np.array,
    Y_columns: Dict[str, int],
    batch_size: int = 512,
    return_model: bool = False,
    model: str = 'deepeye',
    feature: str = 'Sac',
    optimizer: str = 'adam_w',
    use_data_generator: bool = True,
    epochs: int = 20,
    patience: int = 50,
):
    print('evaluate and create test embeddings')
    if model == 'deepeye':
        # clear tensorflow session
        tf.keras.backend.clear_session()

        if batch_size != -1:
            model_conf['batch_size'] = batch_size

        # load  model configuration
        conf = configuration.load_config(model_conf)

        # encode label
        le = LabelEncoder()
        le.fit(Y_train[:, Y_columns['subId']])
        Y_train[:, Y_columns['subId']] = le.transform(
            Y_train[:, Y_columns['subId']],
        )

        # one-hot-encode user ids:
        n_train_users_f = len(np.unique(Y_train[:, Y_columns['subId']]))
        y_train = to_categorical(
            Y_train[:, Y_columns['subId']], num_classes=n_train_users_f,
        )

        # SET UP PARAMS FOR NN

        # calculate z-score normalization for fast subnet and add it to configuration:
        m = np.mean(X_train[:, :, [0, 1]], axis=None)
        sd = np.std(X_train[:, :, [0, 1]], axis=None)

        conf.subnets[1].normalization = conf.subnets[1].normalization._replace(
            mean=m, std=sd,
        )

        seq_len = X_train.shape[1]
        n_channels = X_train.shape[2]
        n_classes = y_train.shape[1]

        X_diffs_train = X_train[:, :, [0, 1]] - X_train[:, :, [2, 3]]
        mean_vel_diff = np.nanmean(X_diffs_train)
        std_vel_diff = np.nanstd(X_diffs_train)

        X_diffs_test = X_test[:, :, [0, 1]] - X_test[:, :, [2, 3]]

        # TRAIN NN
        deepeye = deepeyedentification_tf2.DeepEyedentification2Diffs(
            conf.subnets[0],
            conf.subnets[1],
            conf,
            seq_len=seq_len,
            channels=n_channels,
            n_classes=n_classes,
            zscore_mean_vel_diffs=mean_vel_diff,
            zscore_std_vel_diffs=std_vel_diff,
        )

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        for train_index, val_index in skf.split(X_train, Y_train[:, Y_columns['subId']]):
            break

        print('Training model on train data ...')

        # train the model
        cur_hist = deepeye.train(
            X_train, X_diffs_train, y_train,
            train_idx=train_index, validation_idx=val_index,
        )
        print('done.')

        from tensorflow.keras import Model
        print('Creating embedding for test data ...')
        embedding_fast_subnet = Model(
            inputs=deepeye.fast_subnet.input,
            outputs=deepeye.fast_subnet.get_layer('fast_d3').output,
        )
        embedding_slow_subnet = Model(
            inputs=deepeye.slow_subnet.input,
            outputs=deepeye.slow_subnet.get_layer('slow_d3').output,
        )
        embedding_deepeye = Model(
            inputs=deepeye.model.input,
            outputs=deepeye.model.get_layer('deepeye_a2').output,
        )

        embeddings_fast_subnet_all = embedding_fast_subnet.predict(
            [X_test, X_diffs_test],
        )
        embeddings_slow_subnet_all = embedding_slow_subnet.predict(
            [X_test, X_diffs_test],
        )
        embeddings_deepeye_all = embedding_deepeye.predict(
            [[X_test, X_diffs_test], [X_test, X_diffs_test]],
        )

        embeddings_concatenated_all = np.hstack([
            embeddings_fast_subnet_all,
            embeddings_slow_subnet_all,
            embeddings_deepeye_all,
        ])
        print('done.')
        if return_model:
            return embeddings_concatenated_all, deepeye
        else:
            return embeddings_concatenated_all
    elif model == 'lohr':
        tf.keras.backend.clear_session()

        # encode label
        le = LabelEncoder()
        le.fit(Y_train[:, Y_columns['subId']])
        Y_train[:, Y_columns['subId']] = le.transform(
            Y_train[:, Y_columns['subId']],
        )

        # one-hot-encode user ids:
        n_train_users_f = len(np.unique(Y_train[:, Y_columns['subId']]))
        y_train = to_categorical(
            Y_train[:, Y_columns['subId']], num_classes=n_train_users_f,
        )

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience),
            TerminateOnNaN(),
        ]

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        for train_index, val_index in skf.split(X_train, Y_train[:, Y_columns['subId']]):
            break

        print('Training model on train data ...')

        # train the model
        X_train_lohr = X_train[train_index, :]
        y_train_lohr = Y_train[train_index, :]
        X_val_lohr = X_train[val_index, :]
        y_val_lohr = Y_train[val_index, :]
        X_train_lohr = np.nan_to_num(X_train_lohr, nan=0)
        X_val_lohr = np.nan_to_num(X_val_lohr, nan=0)

        if use_data_generator:
            lr = 0.0001
            for xx in range(1000):
                tf.keras.backend.clear_session()
                # TRAIN NN
                model_lohr = lohr_model.build_lohr_model(
                    FEATURE_INPUT_SIZE_DICT[feature],
                    optimizer,
                    lr=lr,
                )
                num_examples_per_epoch = 200000
                num_examples_per_validation = int(
                    y_val_lohr.shape[0] / batch_size,
                )
                steps_per_epoch = int(num_examples_per_epoch / batch_size)
                cur_hist = model_lohr.fit(
                    lohr_model.generate_data(
                        X_train_lohr, y_train_lohr[:, Y_columns['subId']],
                    ),
                    validation_data=lohr_model.generate_data(
                        X_val_lohr, y_val_lohr[:, Y_columns['subId']],
                    ),
                    # validation_data=(X_val_lohr,y_val_lohr[:,Y_columns['subId']]),
                    epochs=epochs, batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=num_examples_per_validation,
                    callbacks=callbacks,
                )
                if np.isnan(cur_hist.history['val_loss'][-1]):
                    print('train one more time')
                    lr = lr * 0.9
                else:
                    break
        else:
            lr = 0.0001
            for xx in range(1000):
                tf.keras.backend.clear_session()
                # TRAIN NN
                model_lohr = lohr_model.build_lohr_model(
                    FEATURE_INPUT_SIZE_DICT[feature],
                    optimizer,
                    lr=lr,
                )

                cur_hist = model_lohr.fit(
                    x=X_train_lohr, y=y_train_lohr[:, Y_columns['subId']],
                    validation_data=(
                        X_val_lohr, y_val_lohr[:, Y_columns['subId']],
                    ),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=callbacks,
                    shuffle=True,
                )
                if np.isnan(cur_hist.history['val_loss'][-1]):
                    print('train one more time')
                    lr = lr * 0.9
                else:
                    break
        print('done.')

        from tensorflow.keras import Model
        print('Creating embedding for test data ...')
        embeddings_lohr = model_lohr.predict(
            X_test,
        )

        print('done.')
        if return_model:
            return embeddings_lohr, model_lohr
        else:
            return embeddings_lohr
