# Toronto Weather Forecasting

## Purpose

This project was part of my [python-weather](https://github.com/james-wukong/python-weather) project, but I think it would be clear to create a new repository for weather forecasting in Toronto. The main purpose of this project is to have some in-hand practises in deep learning and tensorflow framework.

## Dataset

The dataset was collected from [Visual Crossing](https://www.visualcrossing.com), it includes more than 3 years of hourly based recorded observations and this project only focuses on the weather in Toronto.

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
6. train the model in LSTM neural networks
7. evaluate by comparing loss in training and validation processes
8. evaluate by testing dataset predictions
9. fine turning hyperparameters to get a better result


