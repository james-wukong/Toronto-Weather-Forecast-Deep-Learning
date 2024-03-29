import os

import pandas as pd
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
    n_steps_in, n_steps_out = int(cfg['param']['n_step_in']), int(
        cfg['param']['n_step_out'])
    # training, validation and testing portions
    train_portion, valid_portion = float(cfg['param']['train_portion']), float(
        cfg['param']['valid_portion'])
    # batch size, epochs and learning rates wrt model
    batch_size, epochs = int(cfg['param']['batch_size']), int(
        cfg['param']['epochs'])
    learning_rate = float(cfg['param']['learning_rate'])
    # load the data from dir
    build_print_line('start preparing data')
    df = load_df_from_dir(cfg['data']['history_weather'])
    label_columns = defaultdict(tuple, {
        'reg': ('temp', 'feelslike'),
        'cls': ('ohe_rain', 'ohe_snow')
    })

    director = Director()
    builder = ConcreteBuilderWeather(df)
    director.builder = builder

    # data preprocessing
    director.build_weather_dataset()

    # get datasets: after scaling, batching, and training dataset shuffling
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
    model = LstmBasedModel(n_steps_in=n_steps_in,
                           n_features=n_features,
                           n_steps_out=n_steps_out,
                           n_features_reg_out=len(label_columns['reg']),
                           n_features_cls_out=len(label_columns['cls']),
                           default_units=1024)

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
    monitor_dict = defaultdict(str, {
        'reg_out': 'reg_out_loss',
        'cls_out': 'cls_out_accuracy'
    })
    output_monitors = defaultdict(str, {
        'reg_out': 'reg_out_loss',
        'cls_out': 'cls_out_accuracy'
    })

    build_print_line(f'start predicting the next {n_steps_out} hours:\n')
    predictions = model.predict(builder.predict_input)
    predictions_reg_inverse = builder.inverse_label_sequence(
        predictions['reg_out'], label_columns=label_columns['reg'])
    predictions_reg = builder.unscale_prediction(n_steps_in,
                                                 predictions_reg_inverse,
                                                 label_columns)
    # print('predictions reg: ', predictions['reg_out'])
    # print('predictions cls: ', predictions['cls_out'])

    binary_values = (predictions['cls_out']
                     >= cfg['param']['threshold']).astype(int)
    print('weather in prevous 10 hours: \n',
          builder.predict_input_original.iloc[-12:, :].to_string(index=False))
    result = np.concatenate((predictions_reg, np.array(binary_values[0])),
                            axis=1)
    result = pd.DataFrame(result, columns=['temp', 'feelslike', 'rain', 'snow'])
    print('predicted results in next 6 hours: \n ',
          result.to_string(index=False))
