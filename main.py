import os
# import logging
import yaml
from collections import defaultdict
import pandas as pd
from src.weather import *
from src.helpers import load_df_from_dir, missing_values
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from src.weather_nn import LSTMLikeModel
from src.weather_nn import PrintEpochProgress, EarlyStoppingAtMinLoss, LossAndErrorLoggingCallback

# Configure logging
# logging.basicConfig(level=logging.DEBUG, filename='weather.log', filemode='w')
# Debug Configuration
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# ignore some output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # GPU device
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

if __name__ == '__main__':
    with open('settings.yaml') as setting:
        cfg = yaml.safe_load(setting)

    # prod_config
    # n_step_in and n_step_out: using timesteps in history to predict timesteps in future
    INPUT_WIN_SIZE, OUTPUT_WIN_SIZE = 24 * 7, 24 * 1
    # training, validation and testing portions
    TRAIN_PORTION, VALID_PORTION = 0.8, 0.1
    # batch size, epochs and learning rates wrt model
    BATCH_SIZE, EPOCHS, LR = 24 * 90, 100, 5e-3
    # load the data from dir
    df = load_df_from_dir(cfg['data']['history_weather'])
    LABEL_COLUMNS = defaultdict(tuple, {'reg': ('temp', 'feelslike'), 'cls': ('ohe_rain', 'ohe_snow')})

    # test config
    # INPUT_WIN_SIZE, OUTPUT_WIN_SIZE = 24 * 1, 24 * 1
    # TRAIN_PORTION, VALID_PORTION = 0.8, 0.1
    # BATCH_SIZE, EPOCHS, LR = 24 * 3, 1, 5e-3
    # test data
    # df = pd.read_csv(os.path.join(cfg['data']['history_weather'],
    #                               'Toronto,Ontario,CA 2020-12-11 to 2021-01-20.csv'))
    # plt.plot(df['datetime'], df['temp'])
    # plt.show()
    director = Director()
    builder = ConcreteBuilderWeather(df)
    director.builder = builder

    # data preprocessing
    director.build_weather_dataset()
    # check missing values and make sure data is clean now
    missing_values(director.builder.weather.df)
    print(director.builder.weather.df.columns)

    # selected_feats = ['temp', 'feelslike', 'dew', 'snowdepth', 'windgust',
    #                   'humidity', 'precip', 'precipprob', 'snow',
    #                   'windspeed', 'winddir', 'sealevelpressure',
    #                   'cloudcover', 'visibility', 'solarradiation',
    #                   'severerisk', 'day', 'month', 'year',
    #                   'dayofweek', 'weekofyear', 'hour', 'season']
    # selected_feats_indices = builder.weather.df.columns.get_indexer(selected_feats)

    # get datasets: after scaling, batching, and training dataset shuffling
    train_dataset, val_dataset, test_dataset = builder.create_train_test(
        input_win_size=INPUT_WIN_SIZE,
        output_win_size=OUTPUT_WIN_SIZE,
        train_size=TRAIN_PORTION,
        validate_size=VALID_PORTION,
        batch_size=BATCH_SIZE,
        label_columns=LABEL_COLUMNS,
    )

    n_steps_in, n_steps_out, n_features = INPUT_WIN_SIZE, OUTPUT_WIN_SIZE, len(director.builder.weather.df.columns)

    model = LSTMLikeModel(n_steps_in=n_steps_in, n_features=n_features,
                          n_steps_out=n_steps_out, n_features_reg_out=len(LABEL_COLUMNS['reg']),
                          n_features_cls_out=len(LABEL_COLUMNS['cls']), default_units=512)

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=LR),
        loss={
            'reg_out': keras.losses.MeanSquaredError(),
            'cls_out': keras.losses.BinaryCrossentropy(),
        },
        metrics={
            'reg_out': ['mae'],
            'cls_out': ['accuracy'],
        },
        weighted_metrics=[],

    )

    # define the checkpoint callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=cfg['saved_models']['chkpoint_model'],  # Path to save the model
        monitor='val_loss',  # Metric to monitor (e.g., validation loss)
        save_best_only=True,  # Save only the best models based on the monitored metric
        save_weights_only=True,  # Save the entire model, not just the weights
        mode='min',  # 'min' means save the model when the monitored metric is minimized
        verbose=2,  # Display more information about the saving process
    )

    # train the model
    train_element = next(iter(train_dataset))
    val_element = next(iter(train_dataset))
    test_element = next(iter(test_dataset))
    X_train, y_train_reg, y_train_cls = train_element[0].numpy(), train_element[1].numpy(), train_element[2].numpy()
    X_val, y_val_reg, y_val_cls = val_element[0].numpy(), val_element[1].numpy(), val_element[2].numpy()
    X_test, y_test_reg, y_test_cls = test_element[0].numpy(), test_element[1].numpy(), test_element[2].numpy()

    history = model.fit(X_train,
                        {
                            'reg_out': y_train_reg,
                            'cls_out': y_train_cls,
                        },
                        epochs=EPOCHS,
                        validation_data=(
                            X_val,
                            {
                                'reg_out': y_val_reg,
                                'cls_out': y_val_cls,
                            }
                        ),
                        callbacks=[
                            checkpoint_callback,
                            PrintEpochProgress(),
                            EarlyStoppingAtMinLoss(patience=6, restore_best_weights=True),
                            LossAndErrorLoggingCallback(),
                        ],
                        shuffle=True,
                        verbose=2,
                        )
    model.summary()
    # Plot the training and validation loss
    builder.weather.display_losses(history=history)
    #
    # # evaluate the model
    test_loss = model.evaluate(X_test,
                               {
                                   'reg_out': y_test_reg,
                                   'cls_out': y_test_cls,
                               },
                               verbose=2,
                               sample_weight=None, )
    print(f'Test Loss: {test_loss}')
    # model.save_weights(cfg['saved_models']['weights_model'])
    #
    # # Get the shape of the dataset
    # dataset_shape = tf.data.experimental.get_structure(test_dataset)
    # # Get an iterator for the dataset, Get the first element from the dataset
    # test_element = next(iter(test_dataset))
    #
    # # Extract values directly
    # X_test, y_test_reg, y_test_cls = test_element[0].numpy(), test_element[1].numpy(), test_element[2].numpy()
    #
    # # Predict on the test set
    # y_test_hat = model.predict(X_test)
    # y_test_reg_hat_inverse = builder.inverse_label_sequence(y_test_hat['reg_out'], 24, 24,
    #                                                         label_columns=LABEL_COLUMNS['reg'])
    # y_test_reg_cls_inverse = builder.inverse_label_sequence(y_test_hat['cls_out'], 24, 24,
    #                                                         label_columns=LABEL_COLUMNS['cls'])
    # print('y_test_hat:', y_test_reg_hat_inverse.shape, type(y_test_reg_hat_inverse))
    #
    # print(y_test.shape, y_test_hat.shape)
    # #
    # # # Plot actual vs predicted values\
    # # invert predictions
    # print(type(builder.scaler))
    # y_train_hat = builder.scaler.inverse_transform(y_train_hat)
    # y_train = builder.scaler.inverse_transform([y_train])
    # y_val_hat = builder.scaler.inverse_transform(y_val_hat)
    # y_val = builder.scaler.inverse_transform([y_val])
    # y_test_hat = builder.scaler.inverse_transform(y_test_hat)
    # y_test = builder.scaler.inverse_transform([y_test])

    # Inverse transform to get original scale
    # print(y_test.shape, type(y_test))
    # print(builder.y_test_ds.shape, type(builder.y_test_ds))
    # assert np.array_equal(y_test, builder.y_test_ds), 'y_test not equal to builder_y_test'
    # true_values = builder.scaler.inverse_transform(y_test_hat.reshape(-1, 3))
    # predicted_values = builder.scaler.inverse_transform(y_test_hat.flatten().reshape(-1, 3))
    # print(true_values.shape)
    # print(predicted_values.shape)
    #
    # builder.weather.plot_test_predictions(true_values, predicted_values)
