# import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Any
import logging


class MultiOutputModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor_dict, mode='min', **kwargs):
        super(MultiOutputModelCheckpoint, self).__init__(filepath, mode, **kwargs)
        self.filepath = filepath
        self.monitor_dict = monitor_dict
        self.mode = mode
        self.best_metrics = {output: float('inf') if mode == 'min' else float('-inf') for output in monitor_dict}

    def on_epoch_end(self, epoch, logs=None):
        print('in checkpoints custom logs: ', logs)
        for output, monitor_metric in self.monitor_dict.items():
            current_metric = logs.get(monitor_metric)
            if current_metric is None:
                raise ValueError(f"Metric {monitor_metric} is not available in the training logs.")

            if (self.mode == 'min' and current_metric < self.best_metrics[output]) or \
                    (self.mode == 'max' and current_metric > self.best_metrics[output]):

                self.best_metrics[output] = current_metric

                if self.save_best_only and self.save_weights_only:
                    # self.model.save(filepath=self.filepath.format(output=output, epoch=epoch + 1), overwrite=True)
                    self.model.save_weights(filepath=self.filepath.format(output=output, epoch=epoch + 1),
                                            overwrite=True)

            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {output} improved from {self.best_metrics[output]:.4f} "
                      f"to {current_metric:.4f}, saving model: {self.filepath.format(output=output, epoch=epoch + 1)}")


class PrintEpochProgress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        print info at the end of each epoch
        """
        print('\nEpoch:', epoch + 1, ' Loss:', logs['loss'])


class EarlyStoppingAtMinLoss(keras.callbacks.EarlyStopping):
    """
    Early stop when loss does not improve over epochs
    """

    def __init__(self, monitor: str = "val_loss", patience: int = 0, restore_best_weights: bool = False):
        super(EarlyStoppingAtMinLoss, self).__init__(monitor=monitor, patience=patience,
                                                     restore_best_weights=restore_best_weights)
        self.best = None
        self.patience = patience
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # combined_loss = sum(logs[output] for output in self.monitor_dict.values())
        # logs['combined_loss'] = sum(logs[output] for output in self.monitor_dict.values())
        # Perform the regular early stopping check
        super(EarlyStoppingAtMinLoss, self).on_epoch_end(epoch, logs)

        current = logs.get(self.monitor)
        # current = logs.get('combined_loss')
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


class LossAndErrorLoggingCallback(keras.callbacks.Callback):
    """
    log loss and errors
    """

    def on_train_batch_end(self, batch, logs=None):
        logging.debug('Up to batch {}, the average loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_test_batch_end(self, batch, logs=None):
        logging.debug('Up to batch {}, the average loss is {:7.2f}.'.format(batch, logs['loss']))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        metrics = logs['mean_absolute_error'] if 'mean_absolute_error' in keys else 0
        logging.debug('The average loss for epoch {} is {:7.2f} '.format(epoch, logs['loss']))
        logging.debug('and mean absolute error is {:7.2f}.'.format(metrics))


class LstmBasedModel(keras.Model):
    """
    Custom LSTM Based NN Model
    """

    def __init__(self, n_steps_in, n_features, n_steps_out,
                 n_features_reg_out, n_features_cls_out, default_units=512):
        super(LstmBasedModel, self).__init__()
        self.n_steps_in = n_steps_in
        self.n_features = n_features
        self.n_steps_out = n_steps_out
        self.n_features_reg_out = n_features_reg_out
        self.n_features_cls_out = n_features_cls_out
        self.default_units = default_units

        self.input_layer = layers.LSTM(default_units, activation='tanh',
                                       input_shape=(n_steps_in, n_features),
                                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                                       # return_sequences=True,
                                       )
        self.repvect_layer = layers.RepeatVector(n_steps_out)
        self.bn_1 = layers.BatchNormalization()
        self.dropout_1 = layers.Dropout(0.25, name='dropout_1')

        self.lstm_layer_bi = layers.Bidirectional(
            layers.LSTM(256,
                        return_sequences=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
        self.bn_2 = layers.BatchNormalization()
        self.dropout_2 = layers.Dropout(0.25, name='dropout_2')

        self.lstm_layer_bi3 = layers.Bidirectional(
            layers.LSTM(62,
                        return_sequences=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
        self.bn_3 = layers.BatchNormalization()
        self.dropout_3 = layers.Dropout(0.25, name='dropout_3')

        self.timed_layer_reg = layers.TimeDistributed(layers.Dense(n_features_reg_out), name='reg_out')
        self.timed_layer_cls = layers.TimeDistributed(layers.Dense(n_features_cls_out,
                                                                   activation='sigmoid'), name='cls_out')

    def call(self, inputs, training=False, mask: Any = None) -> Any:
        x = self.input_layer(inputs)
        x = self.repvect_layer(x)
        x = self.bn_1(x, training=training)
        x = self.dropout_1(x, training=training)

        x = self.lstm_layer_bi(x)
        x = self.bn_2(x, training=training)
        x = self.dropout_2(x, training=training)

        x = self.lstm_layer_bi3(x)
        x = self.bn_3(x, training=training)
        x = self.dropout_3(x, training=training)

        output_reg = self.timed_layer_reg(x)
        output_cls = self.timed_layer_cls(x)
        return {'reg_out': output_reg, 'cls_out': output_cls}

    def build(self, input_shape):
        """
        to correctly display summary() of the model
        """
        x = keras.Input(shape=(self.n_steps_in, self.n_features,))

        return keras.Model(inputs=[x], outputs=self.call(x))


class GruBasedModel(keras.Model):
    """
    Custom GRU Based NN Model
    """

    def __init__(self, n_steps_in, n_features, n_steps_out,
                 n_features_reg_out, n_features_cls_out, default_units=512):
        super(GruBasedModel, self).__init__()
        self.n_steps_in = n_steps_in
        self.n_features = n_features
        self.n_steps_out = n_steps_out
        self.n_features_reg_out = n_features_reg_out
        self.n_features_cls_out = n_features_cls_out
        self.default_units = default_units

        self.input_layer = layers.GRU(default_units, activation='tanh',
                                       input_shape=(n_steps_in, n_features),
                                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                                       # return_sequences=True,
                                       )
        self.repvect_layer = layers.RepeatVector(n_steps_out)
        self.bn_1 = layers.BatchNormalization()
        self.dropout_1 = layers.Dropout(0.25, name='dropout_1')

        self.gru_layer_bi = layers.Bidirectional(
            layers.GRU(256,
                        return_sequences=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
        self.bn_2 = layers.BatchNormalization()
        self.dropout_2 = layers.Dropout(0.25, name='dropout_2')

        self.gru_layer_bi3 = layers.Bidirectional(
            layers.GRU(62,
                        return_sequences=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
        self.bn_3 = layers.BatchNormalization()
        self.dropout_3 = layers.Dropout(0.25, name='dropout_3')

        self.timed_layer_reg = layers.TimeDistributed(layers.Dense(n_features_reg_out), name='reg_out')
        self.timed_layer_cls = layers.TimeDistributed(layers.Dense(n_features_cls_out,
                                                                   activation='sigmoid'), name='cls_out')

    def call(self, inputs, training=False, mask: Any = None) -> Any:
        x = self.input_layer(inputs)
        x = self.repvect_layer(x)
        x = self.bn_1(x, training=training)
        x = self.dropout_1(x, training=training)

        x = self.gru_layer_bi(x)
        x = self.bn_2(x, training=training)
        x = self.dropout_2(x, training=training)

        x = self.gru_layer_bi3(x)
        x = self.bn_3(x, training=training)
        x = self.dropout_3(x, training=training)

        output_reg = self.timed_layer_reg(x)
        output_cls = self.timed_layer_cls(x)
        return {'reg_out': output_reg, 'cls_out': output_cls}

    def build(self, input_shape):
        """
        to correctly display summary() of the model
        """
        x = keras.Input(shape=(self.n_steps_in, self.n_features,))

        return keras.Model(inputs=[x], outputs=self.call(x))


class BaseModel(keras.Model):
    """
    Custom Base NN Model
    """

    def __init__(self, n_steps_in, n_features, n_steps_out,
                 n_features_reg_out, n_features_cls_out,
                 default_units=512):
        super(BaseModel, self).__init__()
        self.n_steps_in = n_steps_in
        self.n_features = n_features
        self.n_steps_out = n_steps_out
        self.n_features_reg_out = n_features_reg_out
        self.n_features_cls_out = n_features_cls_out
        self.default_units = default_units

        self.input_layer = layers.LSTM(default_units, activation='tanh',
                                       input_shape=(n_steps_in, n_features),
                                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                                       )
        # self.bn_1 = layers.BatchNormalization()
        self.repvect_layer = layers.RepeatVector(n_steps_out)
        self.dropout_1 = layers.Dropout(0.25, name='dropout_1')

        self.timed_layer_reg = layers.TimeDistributed(layers.Dense(n_features_reg_out), name='reg_out')
        self.timed_layer_cls = layers.TimeDistributed(layers.Dense(n_features_cls_out,
                                                                   activation='sigmoid'), name='cls_out')

    def call(self, inputs, training=False, mask: Any = None) -> Any:
        x = self.input_layer(inputs)
        # x = self.bn_1(x, training=training)
        x = self.repvect_layer(x)
        x = self.dropout_1(x, training=training)

        output_reg = self.timed_layer_reg(x)
        output_cls = self.timed_layer_cls(x)

        return {'reg_out': output_reg, 'cls_out': output_cls}

    def build(self, input_shape):
        """
        to correctly display summary() of the model
        """
        x = keras.Input(shape=(self.n_steps_in, self.n_features,))

        return keras.Model(inputs=[x], outputs=self.call(x))
