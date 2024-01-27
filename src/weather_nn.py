# import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Any
import logging

logging.basicConfig(level=logging.DEBUG, filename='weather_nn.log', filemode='a')


class PrintEpochProgress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        print info at the end of each epoch
        """
        print('\nEpoch:', epoch, ' Loss:', logs['loss'])


class EarlyStoppingAtMinLoss(keras.callbacks.EarlyStopping):
    """
    Early stop when loss does not improve over epochs
    """
    def __init__(self, patience: int = 0, restore_best_weights: bool = False):
        super(EarlyStoppingAtMinLoss, self).__init__(patience=patience,
                                                     restore_best_weights=restore_best_weights)
        self.best = None
        self.patience = patience
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
        # Perform the regular early stopping check
        super(EarlyStoppingAtMinLoss, self).on_epoch_end(epoch, logs)

        current = logs.get('loss')
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
        metrics = logs['mean_absolute_error'] if 'mean_absolute_error' in keys else None
        logging.debug('The average loss for epoch {} is {:7.2f} ')
        logging.debug('and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], metrics))


class LSTMLikeModel(keras.Model):
    """
    Custom LSTM Like Model
    """

    def __init__(self, n_steps_in, n_features, n_steps_out, n_features_out, default_units=512):
        super(LSTMLikeModel, self).__init__()
        self.n_steps_in = n_steps_in
        self.n_features = n_features
        self.n_steps_out = n_steps_out
        self.n_features_out = n_features_out
        self.default_units = default_units

        self.input_layer = layers.LSTM(default_units, activation='softmax',
                                       input_shape=(n_steps_in, n_features),
                                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                                       # return_sequences=True,
                                       )
        self.dropout = layers.Dropout(0.5)
        # self.flatten_layer = layers.Flatten()
        # self.reshape_layer = layers.Reshape((n_steps_in * default_units,))
        self.repvect_layer = layers.RepeatVector(n_steps_out)
        # self.lstm_layer = layers.LSTM(256, activation='softmax',
        #                               return_sequences=True,
        #                               kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))
        self.lstm_layer_bi = layers.Bidirectional(
            layers.LSTM(256,
                        return_sequences=True,
                        kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
        self.timed_layer = layers.TimeDistributed(layers.Dense(n_features_out))

    def call(self, inputs, training=False, mask: Any = None) -> Any:
        x = self.input_layer(inputs)
        # x = self.flatten_layer(x)
        # x = self.reshape_layer(x)
        x = self.repvect_layer(x)
        # x = self.dropout(x)
        # x = self.lstm_layer(x)
        x = self.dropout(x)
        x = self.lstm_layer_bi(x)
        # x = self.dropout(x)

        output = self.timed_layer(x)
        return output

    def build(self, input_shape):
        """
        to correctly display summary() of the model
        """
        x = keras.Input(shape=(self.n_steps_in, self.n_features,))

        return keras.Model(inputs=[x], outputs=self.call(x))
