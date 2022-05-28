import os.path
import random
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from tensorflow import Variable
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.math import l2_normalize
from tensorflow_addons.losses import TripletSemiHardLoss
from tensorflow_addons.optimizers import AdamW


# Create batch with 100 (20 subjects * 5 events each) instances for metric learning
def create_batch(
    X: np.array, y: np.array, num_subjects: int = 20, num_events: int = 5, batchsize: int = 100,
) -> Tuple[np.array, np.array]:
    unique_ids = np.array(list(np.unique(y)))
    random.shuffle(unique_ids)
    use_subs = unique_ids[0:num_subjects]
    use_ids = []
    for i in range(len(use_subs)):
        cur_ids = np.where(y == use_subs[i])[0]
        random.shuffle(cur_ids)
        use_ids += list(cur_ids[0:num_events])
    if len(use_ids) < batchsize:
        num_add = batchsize - len(use_ids)
        print(num_add)
        rand_ids = np.arange(len(y))
        random.shuffle(rand_ids)
        use_ids += list(rand_ids[0:num_add])
    X_out = X[use_ids]
    y_out = y[use_ids]
    return X_out, y_out


# Load data
def generate_data(
    X: np.array, y: np.array, num_subjects: int = 20, num_events: int = 5, batchsize: int = 100,
) -> Generator[Tuple[np.array, np.array], None, None]:
    while True:
        X, y = create_batch(X, y, num_subjects=20, num_events=5, batchsize=100)
        yield (X, y)


# Build model
def build_lohr_model(
    input_size: Union[int, Tuple[int, ...]],
    optimizer: Optional[str] = None,
    lr: float = 0.0001,
) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_size),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64),
            Lambda(lambda x: l2_normalize(x, axis=1), name='embedding_layer'),
        ],
    )

    # define steps and schedule for AdamW
    step = Variable(0, trainable=False)
    schedule = PiecewiseConstantDecay(
        [100000, 150000], [1e-3, 1e-4, 1e-5],
    )
    lr = 1e-3 * schedule(step)
    def wd(): return 1e-1 * schedule(step)

    if optimizer is None or optimizer == 'adam_w':
        # adamw
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
    elif optimizer == 'adam':
        # adam
        optimizer = Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=TripletSemiHardLoss(margin=0.2),
    )
    return model


# Get subject list
def get_subjects() -> List[str]:
    base_path = 'lohr_feature_data/'
    print('see README.md for details')
    sub_list = list()
    for sub in os.listdir(base_path):
        sub_list.append(sub.split('_')[1][1:])
    sub_list = sorted(list(set(sub_list)))
    return sub_list


# Load data with Rigas feature
def load_feature_data(
    subjects: List[str],
    task: str = 'TEX',
    rounds: List[int] = list(range(1, 10)),
    event: str = 'Sac',
    sessions: List[int] = [1, 2],

) -> Tuple[np.array, np.array]:
    base_path = 'lohr_feature_data/'
    print('see README.md for details')
    input_size_dict = {
        'Fix': 61,
        'Sac': 81,
        'PSO': 44,
    }
    X = np.empty((0, input_size_dict[event]))
    y = []
    counter = 0
    for sub in subjects:
        sub = sub.zfill(3)
        counter += 1
        for _round in rounds:
            for session in sessions:
                try:
                    tmp_df = pd.read_csv(
                        os.path.join(base_path, f'S_{_round}{sub}_S{session}_{task}/ET_S_{_round}{sub}_S{session}_{task}_Raw_{event}_Features.csv'),  # noqa: E501
                    )
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    continue
                print(f'Loading data for S_{_round}{sub}_S{session}_{task}')
                for row in range(len(tmp_df)):
                    X = np.vstack([X, tmp_df.iloc[row]])
                    y.append(sub)
        if counter == 10:
            break
    X = np.nan_to_num(X, nan=0)
    y = np.array(y)
    return X, y
