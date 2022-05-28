import os
from typing import Tuple

import tensorflow
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import Model


class DeepEyedentification2Diffs():

    def __init__(
        self, config_slow, config_fast, config, seq_len, channels, n_classes,
        zscore_mean_vel_diffs, zscore_std_vel_diffs,
    ):
        self.slow_subnet, self.fast_subnet, self.model = DeepEyedentification2Diffs.build_deepeyedentification(  # noqa: E501
            seq_len,
            channels,
            config_slow,
            config_fast,
            config,
            Nreader=n_classes,
            zscore_mean_vel_diffs=zscore_mean_vel_diffs,
            zscore_std_vel_diffs=zscore_std_vel_diffs,
        )
        self.config_slow = config_slow
        self.config_fast = config_fast
        self.config = config

    def train(
        self, X_vel, X_diff_vel, y, train_idx, validation_idx,
        pretrained_weights_slow_path=None,
        pretrained_weights_fast_path=None,
    ):
        """
        X contains training data (angular gaze velocities), y are training labels (identities)
        X has dimensions n x seq_len, y is n x 1
        validation_idx is boolean array has shape[0] as X, indicates which ids to use for validation
        """

        # callback for early stopping
        callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

        # pretrain slow and fast subnet independently
        if not pretrained_weights_slow_path:
            tf.keras.backend.clear_session()
            print('Train slow subnet...')
            self.slow_subnet.fit(
                [X_vel[train_idx, :], X_diff_vel[train_idx, :]], y[train_idx, :],
                validation_data=(
                    [
                        X_vel[validation_idx, :], X_diff_vel[validation_idx, :],
                    ], y[validation_idx, :],
                ),
                shuffle=True,
                batch_size=self.config_slow.batch_size,
                epochs=self.config.epochs,
                callbacks=callbacks,
            )
            # save weights to be sure:
            self.slow_subnet.save_weights(
                os.path.join(
                    'trained_models', 'weights_{}.h5'.format('slow_subnet'),
                ),
            )

        else:
            print('Loading pretrained weights for slow subnet.')
            self.slow_subnet.load_weights(
                pretrained_weights_slow_path +
                'weights_{}.h5'.format('slow_subnet'),
                by_name=True,
            )

        if not pretrained_weights_fast_path:
            tf.keras.backend.clear_session()
            print('Train fast subnet...')
            self.fast_subnet.fit(
                [X_vel[train_idx, :], X_diff_vel[train_idx, :]], y[train_idx, :],
                validation_data=(
                    [
                        X_vel[validation_idx, :], X_diff_vel[validation_idx, :],
                    ], y[validation_idx, :],
                ),
                shuffle=True,
                batch_size=self.config_fast.batch_size,
                epochs=self.config.epochs,
                callbacks=callbacks,
            )
            # save weights to be sure:
            self.fast_subnet.save_weights(
                os.path.join(
                    'trained_models', 'weights_{}.h5'.format('fast_subnet'),
                ),
            )
        else:
            print('Loading pretrained weights for fast subnet.')
            self.fast_subnet.load_weights(
                pretrained_weights_fast_path +
                'weights_{}.h5'.format('fast_subnet'),
                by_name=True,
            )

        tf.keras.backend.clear_session()
        # load weights of pre-trained subnets into merged net
        for i in range(len(self.slow_subnet.layers)):
            for j in range(len(self.model.layers)):
                if self.model.layers[j].name == self.slow_subnet.layers[i].name:
                    self.model.layers[j].set_weights(
                        self.slow_subnet.layers[i].get_weights(),
                    )

        for i in range(len(self.fast_subnet.layers)):
            for j in range(len(self.model.layers)):
                if self.model.layers[j].name == self.fast_subnet.layers[i].name:
                    self.model.layers[j].set_weights(
                        self.fast_subnet.layers[i].get_weights(),
                    )

        # then train joint architecture with frozen subnet weights
        print('Train merged...')

        history_merged_net = self.model.fit(
            [
                X_vel[train_idx, :], X_diff_vel[train_idx, :],
                X_vel[train_idx, :], X_diff_vel[train_idx, :],
            ],
            y[train_idx, :],
            validation_data=(
                [
                    X_vel[validation_idx, :], X_diff_vel[validation_idx, :],
                    X_vel[validation_idx, :], X_diff_vel[validation_idx, :],
                ],
                y[validation_idx, :],
            ),
            shuffle=True,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=callbacks,
        )

        return history_merged_net

    @staticmethod
    def build_deepeyedentification(
        seq_len, channels, config_slow, config_fast, config_deepeyedentification,
        Nreader, zscore_mean_vel_diffs, zscore_std_vel_diffs,
    ) -> Tuple[Model, Model, Model]:
        concatenate_subnet_dense_layers = True

        # define optimizers from configurations
        opt_deepeyedentification = tensorflow.keras.optimizers.Adam(
            learning_rate=config_deepeyedentification.learning_rate,
        )

        # build slow and fast subnet
        slow_subnet = DeepEyedentification2Diffs.build_slow_subnet(
            seq_len, channels, Nreader=Nreader, config=config_slow,
            zscore_mean_vel_diffs=zscore_mean_vel_diffs,
            zscore_std_vel_diffs=zscore_std_vel_diffs,
        )
        fast_subnet = DeepEyedentification2Diffs.build_fast_subnet(
            seq_len, channels, Nreader=Nreader, config=config_fast,
            zscore_mean_vel_diffs=zscore_mean_vel_diffs,
            zscore_std_vel_diffs=zscore_std_vel_diffs,
        )

        slow_output = slow_subnet.output
        fast_output = fast_subnet.output

        # merge outputs of slow and fast subnet
        if concatenate_subnet_dense_layers:
            slow_bottleneck = slow_subnet.get_layer(name='slow_n12').output
            fast_bottleneck = fast_subnet.get_layer(name='fast_n12').output
            deepeye_merge = Concatenate(axis=1, name='deepeye_merge')(
                [slow_bottleneck, fast_bottleneck],
            )
        else:
            deepeye_merge = Concatenate(axis=1, name='deepeye_merge')([
                slow_output, fast_output,
            ])

        deepeye_d1 = Dense(
            config_deepeyedentification.dense[0], activation=None, name='deepeye_d1',
        )(deepeye_merge)
        deepeye_n1 = BatchNormalization(axis=-1, name='deepeye_n1')(deepeye_d1)
        deepeye_a1 = Activation('relu', name='deepeye_a1')(deepeye_n1)
        deepeye_d2 = Dense(
            config_deepeyedentification.dense[1], activation=None, name='deepeye_d2',
        )(deepeye_a1)
        deepeye_n2 = BatchNormalization(axis=-1, name='deepeye_n2')(deepeye_d2)
        deepeye_a2 = Activation('relu', name='deepeye_a2')(deepeye_n2)
        output_deepeye = Dense(
            Nreader, activation='softmax',
            name='deepeye_sm',
        )(deepeye_a2)

        temp = Model(
            inputs=[
                slow_subnet.inputs[0], slow_subnet.inputs[1],
                fast_subnet.inputs[0], fast_subnet.inputs[1],
            ],
            outputs=output_deepeye,
        )

        # make deep copy of the model (such that layers can be frozen without affecting the subnets)
        deepeye_net = clone_model(temp)

        # freeze weights of subnets
        for layer in deepeye_net.layers[:-8]:
            layer.trainable = False

        # compile merged net
        deepeye_net.compile(
            optimizer=opt_deepeyedentification,
            loss='categorical_crossentropy', metrics=['accuracy'],
        )
        print(deepeye_net.summary())

        return slow_subnet, fast_subnet, deepeye_net

    @staticmethod
    def build_slow_subnet(
        seq_len, channels, Nreader, config, zscore_mean_vel_diffs, zscore_std_vel_diffs,
    ) -> Model:
        # slow subnet architecture

        # velocity diffs input:
        input_diff_velocity = Input(
            shape=(seq_len, 2), name='slow_velocity_diff_input',
        )
        fast_diff_zscore_norm = ZscoreNormalizationLayer(
            zscore_mean=zscore_mean_vel_diffs,
            zscore_std=zscore_std_vel_diffs,
        )(input_diff_velocity)

        # velocity diffs input:
        input_slow = Input(shape=(seq_len, channels), name='slow_input')
        slow_transformation = TransformationLayer(
            transformation=config.transform.transformation,
            tanh_factor=config.transform.factor,
            name='slow_transform',
        )(input_slow)

        # concatenate eye velocity and stimulus velocity input:
        slow_concat1 = Concatenate(axis=-1, name='slow_vel_diff_merge')(
            [slow_transformation, fast_diff_zscore_norm],
        )

        slow_c1 = Conv1D(
            filters=config.filters[0], kernel_size=config.kernel[0], strides=config.strides[0],
            padding='same', kernel_initializer='he_normal', name='slow_c1',
        )(slow_concat1)

        slow_a1 = Activation('relu', name='slow_a1')(slow_c1)
        slow_p1 = AveragePooling1D(
            pool_size=config.pl_size[0], strides=config.pl_strides[0],
            padding='same', name='slow_p1',
        )(slow_a1)
        slow_n1 = BatchNormalization(axis=-1, name='slow_n1')(slow_p1)

        slow_c2 = Conv1D(
            filters=config.filters[1], kernel_size=config.kernel[1], strides=config.strides[1],
            padding='same', kernel_initializer='he_normal', name='slow_c2',
        )(slow_n1)
        slow_a2 = Activation('relu', name='slow_a2')(slow_c2)
        slow_n2 = BatchNormalization(axis=-1, name='slow_n2')(slow_a2)
        slow_p2 = AveragePooling1D(
            pool_size=config.pl_size[1], strides=config.pl_strides[1],
            padding='same', name='slow_p2',
        )(slow_n2)

        slow_c3 = Conv1D(
            filters=config.filters[2], kernel_size=config.kernel[2], strides=config.strides[2],
            padding='same', kernel_initializer='he_normal', name='slow_c3',
        )(slow_p2)
        slow_a3 = Activation('relu', name='slow_a3')(slow_c3)
        slow_n3 = BatchNormalization(axis=-1, name='slow_n3')(slow_a3)
        slow_p3 = AveragePooling1D(
            pool_size=config.pl_size[2], strides=config.pl_strides[2],
            padding='same', name='slow_p3',
        )(slow_n3)

        slow_c4 = Conv1D(
            filters=config.filters[3], kernel_size=config.kernel[3],
            strides=config.strides[3], padding='same',
            kernel_initializer='he_normal', name='slow_c4',
        )(slow_p3)
        slow_a4 = Activation('relu', name='slow_a4')(slow_c4)
        slow_n4 = BatchNormalization(axis=-1, name='slow_n4')(slow_a4)
        slow_p4 = AveragePooling1D(
            pool_size=config.pl_size[3], strides=config.pl_strides[3],
            padding='same', name='slow_p4',
        )(slow_n4)

        slow_c5 = Conv1D(
            filters=config.filters[4], kernel_size=config.kernel[4], strides=config.strides[4],
            padding='same', kernel_initializer='he_normal', name='slow_c5',
        )(slow_p4)
        slow_a5 = Activation('relu', name='slow_a5')(slow_c5)
        slow_n5 = BatchNormalization(axis=-1, name='slow_n5')(slow_a5)
        slow_p5 = AveragePooling1D(
            pool_size=config.pl_size[4], strides=config.pl_strides[4],
            padding='same', name='slow_p5',
        )(slow_n5)

        slow_c6 = Conv1D(
            filters=config.filters[5], kernel_size=config.kernel[5],
            strides=config.strides[5], padding='same',
            kernel_initializer='he_normal', name='slow_c6',
        )(slow_p5)
        slow_a6 = Activation('relu', name='slow_a6')(slow_c6)
        slow_n6 = BatchNormalization(axis=-1, name='slow_n6')(slow_a6)
        slow_p6 = AveragePooling1D(
            pool_size=config.pl_size[5], strides=config.pl_strides[5],
            padding='same', name='slow_p6',
        )(slow_n6)

        slow_c7 = Conv1D(
            filters=config.filters[6], kernel_size=config.kernel[6], strides=config.strides[6],
            padding='same', kernel_initializer='he_normal', name='slow_c7',
        )(slow_p6)
        slow_a7 = Activation('relu', name='slow_a7')(slow_c7)
        slow_n7 = BatchNormalization(axis=-1, name='slow_n7')(slow_a7)
        slow_p7 = AveragePooling1D(
            pool_size=config.pl_size[6], strides=config.pl_strides[6],
            padding='same', name='slow_p7',
        )(slow_n7)

        slow_c8 = Conv1D(
            filters=config.filters[7], kernel_size=config.kernel[7], strides=config.strides[7],
            padding='same', kernel_initializer='he_normal', name='slow_c8',
        )(slow_p7)
        slow_a8 = Activation('relu', name='slow_a8')(slow_c8)
        slow_n8 = BatchNormalization(axis=-1, name='slow_n8')(slow_a8)
        slow_p8 = AveragePooling1D(
            pool_size=config.pl_size[7], strides=config.pl_strides[7],
            padding='same', name='slow_p8',
        )(slow_n8)

        slow_c9 = Conv1D(
            filters=config.filters[8], kernel_size=config.kernel[8],
            strides=config.strides[8], padding='same',
            kernel_initializer='he_normal', name='slow_c9',
        )(slow_p8)
        slow_a9 = Activation('relu', name='slow_a9')(slow_c9)
        slow_n9 = BatchNormalization(axis=-1, name='slow_n9')(slow_a9)
        slow_p9 = AveragePooling1D(
            pool_size=config.pl_size[8], strides=config.pl_strides[8],
            padding='same', name='slow_p9',
        )(slow_n9)

        slow_f = Flatten(name='slow_f')(slow_p9)
        slow_d1 = Dense(
            config.dense[0],
            activation='relu', name='slow_d1',
        )(slow_f)
        slow_n10 = BatchNormalization(axis=-1, name='slow_n10')(slow_d1)

        slow_d2 = Dense(
            config.dense[1], activation='relu', name='slow_d2',
        )(slow_n10)
        slow_n11 = BatchNormalization(axis=-1, name='slow_n11')(slow_d2)

        slow_d3 = Dense(
            config.dense[2], activation='relu', name='slow_d3',
        )(slow_n11)

        slow_n12 = BatchNormalization(axis=-1, name='slow_n12')(slow_d3)

        slow_output = Dense(
            Nreader, activation='softmax',
            name='slow_sm',
        )(slow_n12)

        slow_subnet = Model(
            inputs=[input_slow, input_diff_velocity], outputs=slow_output,
        )

        # compile
        opt_slow = tensorflow.keras.optimizers.Adam(learning_rate=config.learning_rate)
        slow_subnet.compile(
            optimizer=opt_slow, loss='categorical_crossentropy', metrics=['accuracy'],
        )
        return slow_subnet

    @staticmethod
    def build_fast_subnet(
        seq_len, channels, Nreader, config, zscore_mean_vel_diffs, zscore_std_vel_diffs,
    ) -> Model:

        # velocity diffs input:
        input_diff_velocity = Input(
            shape=(seq_len, 2), name='fast_velocity_diff_input',
        )
        fast_diff_zscore_norm = ZscoreNormalizationLayer(
            zscore_mean=zscore_mean_vel_diffs,
            zscore_std=zscore_std_vel_diffs,
        )(input_diff_velocity)

        input_fast = Input(shape=(seq_len, channels), name='fast_input')
        fast_transformation = TransformationLayer(
            transformation=config.transform.transformation,
            clip_threshold=config.transform.threshold, name='fast_transform',
        )(
            input_fast,
        )
        fast_zscore_norm = ZscoreNormalizationLayer(
            zscore_mean=config.normalization.mean,
            zscore_std=config.normalization.std,
        )(fast_transformation)

        # concatenate eye velocity and stimulus velocity input:
        fast_concat1 = Concatenate(axis=-1, name='fast_vel_diff_merge')(
            [fast_zscore_norm, fast_diff_zscore_norm],
        )

        fast_c1 = Conv1D(
            filters=config.filters[0], kernel_size=config.kernel[0], strides=config.strides[0],
            padding='same', kernel_initializer='he_normal', name='fast_c1',
        )(fast_concat1)

        fast_a1 = Activation('relu', name='fast_a1')(fast_c1)
        fast_p1 = AveragePooling1D(
            pool_size=config.pl_size[0], strides=config.pl_strides[0],
            padding='same', name='fast_p1',
        )(fast_a1)
        fast_n1 = BatchNormalization(axis=-1, name='fast_n1')(fast_p1)

        fast_c2 = Conv1D(
            filters=config.filters[1], kernel_size=config.kernel[1], strides=config.strides[1],
            padding='same', kernel_initializer='he_normal', name='fast_c2',
        )(fast_n1)
        fast_a2 = Activation('relu', name='fast_a2')(fast_c2)
        fast_n2 = BatchNormalization(axis=-1, name='fast_n2')(fast_a2)
        fast_p2 = AveragePooling1D(
            pool_size=config.pl_size[1], strides=config.pl_strides[1],
            padding='same', name='fast_p2',
        )(fast_n2)

        fast_c3 = Conv1D(
            filters=config.filters[2], kernel_size=config.kernel[2], strides=config.strides[2],
            padding='same', kernel_initializer='he_normal', name='fast_c3',
        )(fast_p2)
        fast_a3 = Activation('relu', name='fast_a3')(fast_c3)
        fast_n3 = BatchNormalization(axis=-1, name='fast_n3')(fast_a3)
        fast_p3 = AveragePooling1D(
            pool_size=config.pl_size[2], strides=config.pl_strides[2],
            padding='same', name='fast_p3',
        )(fast_n3)

        fast_c4 = Conv1D(
            filters=config.filters[3], kernel_size=config.kernel[3], strides=config.strides[3],
            padding='same', kernel_initializer='he_normal', name='fast_c4',
        )(fast_p3)
        fast_a4 = Activation('relu', name='fast_a4')(fast_c4)
        fast_n4 = BatchNormalization(axis=-1, name='fast_n4')(fast_a4)
        fast_p4 = AveragePooling1D(
            pool_size=config.pl_size[3], strides=config.pl_strides[3],
            padding='same', name='fast_p4',
        )(fast_n4)

        fast_c5 = Conv1D(
            filters=config.filters[4], kernel_size=config.kernel[4], strides=config.strides[4],
            padding='same', kernel_initializer='he_normal', name='fast_c5',
        )(fast_p4)
        fast_a5 = Activation('relu', name='fast_a5')(fast_c5)
        fast_n5 = BatchNormalization(axis=-1, name='fast_n5')(fast_a5)
        fast_p5 = AveragePooling1D(
            pool_size=config.pl_size[4], strides=config.pl_strides[4],
            padding='same', name='fast_p5',
        )(fast_n5)

        fast_c6 = Conv1D(
            filters=config.filters[5], kernel_size=config.kernel[5],
            strides=config.strides[5], padding='same',
            kernel_initializer='he_normal', name='fast_c6',
        )(fast_p5)
        fast_a6 = Activation('relu', name='fast_a6')(fast_c6)
        fast_n6 = BatchNormalization(axis=-1, name='fast_n6')(fast_a6)
        fast_p6 = AveragePooling1D(
            pool_size=config.pl_size[5], strides=config.pl_strides[5],
            padding='same', name='fast_p6',
        )(fast_n6)

        fast_c7 = Conv1D(
            filters=config.filters[6], kernel_size=config.kernel[6], strides=config.strides[6],
            padding='same', kernel_initializer='he_normal', name='fast_c7',
        )(fast_p6)
        fast_a7 = Activation('relu', name='fast_a7')(fast_c7)
        fast_n7 = BatchNormalization(axis=-1, name='fast_n7')(fast_a7)
        fast_p7 = AveragePooling1D(
            pool_size=config.pl_size[6], strides=config.pl_strides[6],
            padding='same', name='fast_p7',
        )(fast_n7)

        fast_c8 = Conv1D(
            filters=config.filters[7], kernel_size=config.kernel[7], strides=config.strides[7],
            padding='same', kernel_initializer='he_normal', name='fast_c8',
        )(fast_p7)
        fast_a8 = Activation('relu', name='fast_a8')(fast_c8)
        fast_n8 = BatchNormalization(axis=-1, name='fast_n8')(fast_a8)
        fast_p8 = AveragePooling1D(
            pool_size=config.pl_size[7], strides=config.pl_strides[7],
            padding='same', name='fast_p8',
        )(fast_n8)

        fast_c9 = Conv1D(
            filters=config.filters[8], kernel_size=config.kernel[8], strides=config.strides[8],
            padding='same', kernel_initializer='he_normal', name='fast_c9',
        )(fast_p8)
        fast_a9 = Activation('relu', name='fast_a9')(fast_c9)
        fast_n9 = BatchNormalization(axis=-1, name='fast_n9')(fast_a9)
        fast_p9 = AveragePooling1D(
            pool_size=config.pl_size[8], strides=config.pl_strides[8],
            padding='same', name='fast_p9',
        )(fast_n9)

        fast_f = Flatten(name='fast_f')(fast_p9)
        fast_d1 = Dense(
            config.dense[0],
            activation='relu', name='fast_d1',
        )(fast_f)
        fast_n10 = BatchNormalization(axis=-1, name='fast_n10')(fast_d1)

        fast_d2 = Dense(
            config.dense[1], activation='relu', name='fast_d2',
        )(fast_n10)
        fast_n11 = BatchNormalization(axis=-1, name='fast_n11')(fast_d2)

        fast_d3 = Dense(
            config.dense[2], activation='relu', name='fast_d3',
        )(fast_n11)
        fast_n12 = BatchNormalization(axis=-1, name='fast_n12')(fast_d3)

        fast_identity_output = Dense(
            Nreader, activation='softmax', name='fast_identity_output',
        )(fast_n12)

        opt_fast = tensorflow.keras.optimizers.Adam(learning_rate=config.learning_rate)

        fast_subnet = Model(
            inputs=[input_fast, input_diff_velocity],
            outputs=fast_identity_output,
        )
        # compile
        fast_subnet.compile(
            optimizer=opt_fast, loss='categorical_crossentropy', metrics=['accuracy'],
        )

        return fast_subnet


class TransformationLayer(Layer):
    def __init__(
        self,
        transformation=None,
        tanh_factor=None,
        clip_threshold=None,
        **kwargs,
    ):
        self.transformation = transformation
        self.tanh_factor = tanh_factor
        self.clip_threshold = clip_threshold
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if self.transformation == 'tanh':
            assert self.tanh_factor is not None, 'Missing tanh factor for tanh transformation.'
            output = tensorflow.keras.backend.tanh(inputs * self.tanh_factor)
            return output

        if self.transformation == 'clip':
            assert self.clip_threshold is not None, 'Missing threshold for clip transformation.'
            if tensorflow.keras.backend.int_shape(inputs)[2] == 2:
                saccade_velocities = tensorflow.keras.backend.sqrt(
                    tensorflow.keras.backend.sum(
                        tensorflow.keras.backend.square(inputs), axis=2, keepdims=True,
                    ),
                )
                condition = tensorflow.keras.backend.cast(
                    tensorflow.keras.backend.greater_equal(
                        saccade_velocities, self.clip_threshold,
                    ), dtype='float32',
                )   # if saccade velocity is below threshold
                condition = tensorflow.keras.backend.repeat_elements(
                    condition, rep=2, axis=2,
                )
                output = inputs * condition
                return output

            if tensorflow.keras.backend.int_shape(inputs)[2] == 4:
                # code for 2 eyes goes here
                saccade_velocities_left = tensorflow.keras.backend.sqrt(
                    tensorflow.keras.backend.square(
                        inputs[:, :, 0],
                    ) + tensorflow.keras.backend.square(inputs[:, :, 1]),
                )
                saccade_velocities_left = tensorflow.keras.backend.expand_dims(
                    saccade_velocities_left, axis=-1,
                )
                saccade_velocities_left = tensorflow.keras.backend.repeat_elements(
                    saccade_velocities_left, rep=2, axis=2,
                )
                condition_left = tensorflow.keras.backend.cast(
                    tensorflow.keras.backend.greater_equal(
                        saccade_velocities_left, self.clip_threshold,
                    ), dtype='float32',
                )

                saccade_velocities_right = tensorflow.keras.backend.sqrt(
                    tensorflow.keras.backend.square(
                        inputs[:, :, 2],
                    ) + tensorflow.keras.backend.square(inputs[:, :, 3]),
                )
                saccade_velocities_right = tensorflow.keras.backend.expand_dims(
                    saccade_velocities_right, axis=-1,
                )
                saccade_velocities_right = tensorflow.keras.backend.repeat_elements(
                    saccade_velocities_right, rep=2, axis=2,
                )
                condition_right = tensorflow.keras.backend.cast(
                    tensorflow.keras.backend.greater_equal(
                        saccade_velocities_right, self.clip_threshold,
                    ), dtype='float32',
                )

                mask = tensorflow.keras.backend.concatenate(
                    (condition_left, condition_right), axis=-1,
                )
                output = inputs * mask
                return output

    def get_config(self):
        config = {
            'transformation': self.transformation,
            'tanh_factor': self.tanh_factor,
            'clip_threshold': self.clip_threshold,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ZscoreNormalizationLayer(Layer):
    def __init__(
        self,
        zscore_mean=None,      # has to be a tensor with same number of channels as input
        zscore_std=None,       # has to be a tensor with same number of channels as input
        **kwargs,
    ):
        self.zscore_mean = zscore_mean
        self.zscore_std = zscore_std
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        assert_message = 'Missing zscore params for normalization.'
        assert self.zscore_mean is not None and self.zscore_std is not None, assert_message
        output = (inputs - self.zscore_mean) / self.zscore_std
        # replace nan with zeros
        value_not_nan = tensorflow.dtypes.cast(
            tensorflow.math.logical_not(
                tensorflow.math.is_nan(output),
            ), dtype=tensorflow.float32,
        )
        output = tensorflow.math.multiply_no_nan(output, value_not_nan)
        return output

    def get_config(self):
        config = {
            'zscore_mean': self.zscore_mean,
            'zscore_std': self.zscore_std,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
