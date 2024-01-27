import os
# import logging
import yaml
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
    # n_step_in and n_step_out: using timesteps in history to predict timesteps in future
    INPUT_WIN_SIZE, OUTPUT_WIN_SIZE = 24 * 7, 24 * 1
    # training, validation and testing portions
    TRAIN_PORTION, VALID_PORTION = 0.8, 0.1
    # batch size, epochs and learning rates wrt model
    BATCH_SIZE, EPOCHS, LR = 24 * 90, 100, 5e-3

    with open('settings.yaml') as setting:
        cfg = yaml.safe_load(setting)

    # load the data from dir
    df = load_df_from_dir(cfg['data']['history_weather'])
    # df = pd.read_csv(os.path.join(cfg['data']['history_weather'],
    #                               'Toronto,Ontario,CA 2020-12-11 to 2021-01-20.csv'))

    director = Director()
    builder = ConcreteBuilderWeather(df)
    director.builder = builder

    # data preprocessing
    director.build_weather_dataset()
    # check missing values and make sure data is clean now
    missing_values(director.builder.weather.df)

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
        label_columns=('temp', 'feelslike', 'ohe_rain'),
    )
    # Regression labels
    # y_regression = weather_data[['temperature', 'humidity', 'wind_speed']]
    #
    # # Classification labels
    # y_classification_rain = (weather_data['rain'] > 0).astype(int)  # 1 if rain, 0 if no rain
    # y_classification_sunny = (weather_data['sunshine'] > 0).astype(int)  # 1 if sunny, 0 if not sunny

    n_features, n_features_out = len(director.builder.weather.df.columns), 3
    n_steps_in, n_steps_out = INPUT_WIN_SIZE, OUTPUT_WIN_SIZE

    model = LSTMLikeModel(n_steps_in=n_steps_in, n_features=n_features,
                          n_steps_out=n_steps_out, n_features_out=n_features_out,
                          default_units=512)

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=LR),
        loss=keras.losses.MeanSquaredError(),
        metrics=['mae'],
    )

    # define the checkpoint callback
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=cfg['saved_models']['chkpoint_model'],  # Path to save the model
        monitor='val_loss',  # Metric to monitor (e.g., validation loss)
        save_best_only=True,  # Save only the best models based on the monitored metric
        save_weights_only=True,  # Save the entire model, not just the weights
        mode='min',  # 'min' means save the model when the monitored metric is minimized
        verbose=2  # Display more information about the saving process
    )

    # train the model
    history = model.fit(train_dataset, epochs=EPOCHS,
                        validation_data=val_dataset,
                        callbacks=[
                            checkpoint_callback,
                            PrintEpochProgress(),
                            EarlyStoppingAtMinLoss(patience=6, restore_best_weights=True),
                            LossAndErrorLoggingCallback(),
                        ],
                        )
    # model.summary()
    # Plot the training and validation loss
    builder.weather.display_losses(history=history)

    # evaluate the model
    test_loss = model.evaluate(test_dataset, verbose=2)
    print(f'Test Loss: {test_loss}')
    # model.save_weights(cfg['saved_models']['weights_model'])
    # Predict on the test set
    # predictions = model.predict(test_dataset)
    #
    # # Plot actual vs predicted values
    # plt.scatter(y_test, predictions)
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')
    # plt.title('Actual vs Predicted Values on Test Set')
    # plt.show()