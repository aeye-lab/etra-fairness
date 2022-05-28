# configuration of deepeyedentification network and its subnets
from collections import namedtuple


class Config_merged_net:
    def __init__(
        self, model_name, subnet1, subnet2, Ndense_merged,
        learning_rate_merged, batch_size_merged=64, epochs=100,
    ):
        # learning_rate_merged:for merged layers
        # batch_size_merged: batch_size for merged layers
        # epochs: number of training epochs for each of the models
        self.model_name = model_name
        self.dense = Ndense_merged
        self.learning_rate = learning_rate_merged
        self.batch_size = batch_size_merged
        self.epochs = epochs
        self.subnets = [subnet1, subnet2]


class Config_subnet:
    def __init__(
        self, subnet_name, transform, normalization, filters, kernel, strides, dense, learning_rate,
        pl_size=[2, 2, 2, 2, 2, 2, 2, 2, 2], pl_strides=[2, 2, 2, 2, 2, 2, 2, 2, 2], batch_size=64,
        input_idx=[0, 1],
    ):
        # transform: Optional tuple containing the transformation name and the respecitve parameters
        #            choose from ('tanh' or 'clip')
        # normalization:  normalization to be applied within each split
        #                 choose from ('zscore' or 'minmax' or None)
        # filters: list containing number of filters for each convolutional layer
        # kernel: list containing kernel size for each convolutional layer
        # strides: list containing number strides for each convolutional layer
        # pl_size: list containing pooling size for each pooling layer
        # pl_strides: list containing pooling strides  for each pooling layer
        # dense: list containing number of nodes for each dense layer
        # learning_rate: learning rate for pretraining
        # batch_size: batch size for pretraining
        # input_idx: list of input feature indeces along third dimension of X.
        #            [0,1]: velocity; [2,3]: acceleration
        assert (
            transform in ('None', None)
            or transform[0] in ('tanh', 'clip')
        )
        if transform in ('None', None):
            self.transform = None
        elif transform[0] == 'tanh':
            TanhTransformation = namedtuple(
                'TanhTransformation', 'transformation factor',
            )
            self.transform = TanhTransformation(
                transformation='tanh', factor=transform[1],
            )
        elif transform[0] == 'clip':
            ClipTransformation = namedtuple(
                'ClipTransformation', 'transformation threshold',
            )
            self.transform = ClipTransformation(
                transformation='clip', threshold=transform[1],
            )
        else:
            self.transform = None

        assert (
            normalization in ('None', None)
            or normalization[0] in ('zscore', 'minmax')
        )
        if normalization in ('None', None):
            self.normalization = None
        else:
            Normalization = namedtuple(
                'Normalization', 'normalization mean std',
            )
            self.normalization = Normalization(
                normalization=normalization[0], mean=None, std=None,
            )
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.pl_size = pl_size
        self.pl_strides = pl_strides
        self.dense = dense
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_idx = input_idx
        self.subnet_name = subnet_name


def load_config(config_json):
    conf_sub1 = Config_subnet(
        subnet_name=config_json['name_sub1'],
        transform=config_json['transform_sub1'],
        normalization=config_json['normalization_sub1'],
        filters=config_json['filters_sub1'],
        kernel=config_json['kernel_sub1'],
        strides=config_json['strides_sub1'],
        dense=config_json['dense_sub1'],
        learning_rate=config_json['learning_rate_sub1'],
        batch_size=config_json['batch_size'],
    )
    conf_sub2 = Config_subnet(
        subnet_name=config_json['name_sub2'],
        transform=config_json['transform_sub2'],
        normalization=config_json['normalization_sub2'],
        filters=config_json['filters_sub2'],
        kernel=config_json['kernel_sub2'],
        strides=config_json['strides_sub2'],
        dense=config_json['dense_sub2'],
        learning_rate=config_json['learning_rate_sub2'],
        batch_size=config_json['batch_size'],
    )
    conf = Config_merged_net(
        model_name=config_json['name_merged'],
        subnet1=conf_sub1,
        subnet2=conf_sub2,
        Ndense_merged=config_json['Ndense_merged'],
        learning_rate_merged=config_json['learning_rate_merged'],
        batch_size_merged=config_json['batch_size'],
    )
    return conf


def load_config_json(config_json_file):
    import json
    fp = open(config_json_file)
    fp_str = fp.read()
    fp.close()
    config_json = json.loads(fp_str)
    return load_config(config_json)


def load_config_from_dataframe(config_df):
    # given a pandas series (one row of df), extract configuration

    n_layers_block1 = 3
    n_layers_block2 = 4
    n_layers_block3 = 2

    conf_sub1 = Config_subnet(
        subnet_name='optimal_slow_subnet',
        transform=['tanh', 20.0],
        normalization=None,

        filters=list(
            int(config_df[f'slow_block{i}_f']) for i in
            [1] * n_layers_block1 + [2] * n_layers_block2 + [3] * n_layers_block3
        ),

        kernel=list(
            int(config_df[f'slow_block{i}_k']) for i in
            [1] * n_layers_block1 + [2] * n_layers_block2 + [3] * n_layers_block3
        ),

        # strides are not tuned:
        strides=[1, 1, 1, 1, 1, 1, 1, 1, 1],

        dense=[int(config_df[f'slow_dense_{i}']) for i in [1, 2, 3]],

        # learning rate is not tuned:
        learning_rate=0.001,
    )

    conf_sub2 = Config_subnet(
        subnet_name='optimal_fast_subnet',
        transform=['clip', 0.04],
        normalization=['zscore'],
        filters=[
            int(config_df[f'fast_block{i}_f']) for i in
            [1] * n_layers_block1 + [2] * n_layers_block2 + [3] * n_layers_block3
        ],
        kernel=[
            int(config_df[f'fast_block{i}_k']) for i in
            [1] * n_layers_block1 + [2] * n_layers_block2 + [3] * n_layers_block3
        ],
        strides=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        dense=[int(config_df[f'fast_dense_{i}']) for i in [1, 2, 3]],
        learning_rate=0.001,
    )
    conf = Config_merged_net(
        model_name='optimal_merged',
        subnet1=conf_sub1,
        subnet2=conf_sub2,
        Ndense_merged=[config_df[f'merged_dense_{i}'] for i in [1, 2]],
        learning_rate_merged=0.00011,
    )

    return conf
