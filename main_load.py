import os
# import logging
import yaml
from tensorflow import keras
from src.weather import *
from src.helpers import load_df_from_dir, build_print_line
from src.weather_nn import LstmBasedModel

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
    n_steps_in, n_steps_out = 24 * 5, int(24 / 4)
    # training, validation and testing portions
    train_portion, valid_portion = 0.70, 0.15
    # batch size, epochs and learning rates wrt model
    batch_size, epochs, learning_rate = 24 * 90, 100, 5e-5
    # load the data from dir
    build_print_line('start loading data from csv files')
    df = load_df_from_dir(cfg['data']['history_weather'])
    label_columns = defaultdict(tuple, {'reg': ('temp', 'feelslike'), 'cls': ('ohe_rain', 'ohe_snow')})

    director = Director()
    builder = ConcreteBuilderWeather(df)
    director.builder = builder

    # data preprocessing
    build_print_line('start preprocessing data')
    director.build_weather_dataset()
    # check missing values and make sure data is clean now
    # missing_values(director.builder.weather.df)
    # print(director.builder.weather.df.columns)

    # get datasets: after scaling, batching, and training dataset shuffling
    build_print_line('start split dataset and split sequences')
    train_dataset, val_dataset, test_dataset = builder.create_train_test(
        input_win_size=n_steps_in,
        output_win_size=n_steps_out,
        train_size=train_portion,
        validate_size=valid_portion,
        batch_size=batch_size,
        label_columns=label_columns,
    )

    n_features = len(director.builder.weather.df.columns)

    build_print_line('start initialize and configure the model')
    model = LstmBasedModel(n_steps_in=n_steps_in, n_features=n_features,
                           n_steps_out=n_steps_out, n_features_reg_out=len(label_columns['reg']),
                           n_features_cls_out=len(label_columns['cls']), default_units=1024)

    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
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

    build_print_line('start loading weights of the model')
    model.load_weights(cfg['saved_models']['chkpoint_model'])
    # model.summary()

    # define the checkpoint callback
    monitor_dict = defaultdict(str, {'reg_out': 'reg_out_loss', 'cls_out': 'cls_out_accuracy'})
    output_monitors = defaultdict(str, {'reg_out': 'reg_out_loss', 'cls_out': 'cls_out_accuracy'})

    # extract datasets from tensor.datasets
    # # Get the shape of the dataset
    # dataset_shape = tf.data.experimental.get_structure(train_dataset)
    train_element = next(iter(train_dataset))
    val_element = next(iter(train_dataset))
    test_element = next(iter(test_dataset))
    X_train, y_train_reg, y_train_cls = train_element[0].numpy(), train_element[1].numpy(), train_element[2].numpy()
    X_val, y_val_reg, y_val_cls = val_element[0].numpy(), val_element[1].numpy(), val_element[2].numpy()
    X_test, y_test_reg, y_test_cls = test_element[0].numpy(), test_element[1].numpy(), test_element[2].numpy()

    # evaluate the model
    build_print_line('start evaluating the model')
    test_loss = model.evaluate(X_test,
                               {
                                   'reg_out': y_test_reg,
                                   'cls_out': y_test_cls,
                               },
                               verbose=2,
                               sample_weight=None, )
    # [overall_loss, cls_out_loss, reg_out_loss, cls_out_accuracy, reg_out_mae]
    # print(f'Test Loss: {test_loss}, type: {type(test_loss)}')

    # # Predict on the test set
    y_test_hat = model.predict(X_test)
    y_test_reg_hat_inverse = builder.inverse_label_sequence(y_test_hat['reg_out'],
                                                            label_columns=label_columns['reg'])
    y_test_reg_inverse = builder.inverse_label_sequence(y_test_reg,
                                                        label_columns=label_columns['reg'])
    y_test_cls_hat_inverse = builder.inverse_label_sequence(y_test_hat['cls_out'],
                                                            label_columns=label_columns['cls'])
    # y_test_cls_inverse = builder.inverse_label_sequence(y_test_cls,
    #                                                     label_columns=label_columns['cls'])

    # plot y_test and y_test_hat
    y_test_hat_reg_unscaled = builder.unscale_prediction(n_steps_in, y_test_reg_hat_inverse, label_columns)
    # y_test_hat_cls_unscaled = builder.inverse_prediction(n_steps_in, y_test_cls_hat_inverse, label_columns)
    y_test_reg_unscaled = builder.unscale_prediction(n_steps_in, y_test_reg_inverse, label_columns)
    WeatherData.plot_test_predictions(y_test_hat_reg_unscaled[:, 0], y_test_reg_unscaled[:, 0])

    # print('y_test_hat:', y_test_reg_hat_inverse.shape, type(y_test_reg_hat_inverse))
    # print('y_test_reg_hat_inverse data: ', y_test_reg_hat_inverse[:10])
    # print('y_test_reg_cls_inverse data: ', y_test_reg_cls_inverse[:10])

