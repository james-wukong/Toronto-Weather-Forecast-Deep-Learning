# Toronto Weather Forecasting

## Dataset

The dataset was collected from [Visual Crossing](https://www.visualcrossing.com), it includes more than 5 years of hourly based recorded observations and this project only focuses on the weather in Toronto.

Thanks for the 1000/day free api from Visual Crossing.

## Overall

It uses weather data in previous days to forecasting the weather conditions in the future hours or days.

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

install libraries listed in requirements.txt

```shell
pip install -r requirements. txt
```

The model is saved by method model.save_weights('data/models/chkpoint_model'), as a result, only weights are saved.

1. to train the model:

```sh
python main_train.py
```

2. to load and evaluate model:

```sh
python main_load.py
```

2. to make predictions with model:

```sh
python main_predict.py
```
