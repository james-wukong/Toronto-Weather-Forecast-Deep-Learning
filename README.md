# Toronto Weather Forecasting

## Dataset

The dataset was collected from [Visual Crossing](https://www.visualcrossing.com), it includes more than 5 years of hourly based recorded observations and this project only focuses on the weather in Toronto.

## Overall

It uses weather data in previous days to forecasting the weather conditions in the future hours or days.

## Project Structure

- data/weather: contains hourly based weather dataset over 5 years
- data/models: contains weights saved from different models
- data/images: learning curve images from models
- data/logs: saved tensor board data
- docs: report for this project
- src: modules and helper function written for this project
- tests: test cases for this project (don't have much time enriching test cases)

## Pipeline

1. read data from csv files into pandas dataframe
2. data cleaning: remove duplicates and handling missing values
3. data preprocessing: scale numerical features and encode categorical features
4. feature engineer: create new features based on the current features, such as 'week of year'
5. create data sequences based on the timestamps and forecast steps
6. split dataset into training, validation and testing datasets
7. all datasets are batched, and only training dataset is shuffled after batching
8. data post-processing: reverse scaled and transformed data to its original values
6. train the model in LSTM neural networks
7. callbacks are used to early stop when loss is not improved anymore
7. evaluate by comparing loss in training and validation processes
8. evaluate by testing dataset predictions
9. fine turning hyperparameters to get a better result

## How To Use

install libraries listed in requirements.txt, make sure you have installed the required libraries before you run the following commands.

```shell
pip install -r requirements. txt
```

The model is saved by method model.save_weights('data/models/chkpoint_model'), as a result, only weights are saved.

Basically all models have been trained, and we don't have to run training again, because it takes hours to complete. However, please feel free to test them and have fun.

We can use 2nd command to load and evaluate the model directly. If you would like to make a prediction, in this case we don't have to input any data, because we are using the last n_step_in (24 * 7) observations to predict n_step_out 6 hours weather. Consequently, just run the 3rd command to make a prediction.

1. to train the model (takes long time):

```sh
python main_train.py
```

1. to load and evaluate model:

```sh
python main_load.py
```

1. to make predictions with model:

```sh
python main_predict.py
```

Additionally, if you are interested in performance of BaseModel or GruBasedModel, you could run the following commands:

```sh
# this will produce the learning curve at the end of evaluation
# evaluate base model
python main_base_load.py

# evaluate gru base model
python main_gru_load.py
```