import numpy as np
import pandas as pd
from src.weather import ConcreteBuilderWeather, WeatherData
import pytest


@pytest.fixture
def multilabel_conditions_columns():
    return [
        'Clear', 'Overcast', 'Partiallycloudy', 'Rain', 'Snow'
    ]


@pytest.fixture
def weather_data():
    return pd.read_csv('../data/weather/Toronto,Ontario,CA 2021-12-15 to 2022-01-24.csv')


@pytest.fixture
def get_builder(weather_data):
    return ConcreteBuilderWeather(weather_data)


@pytest.fixture
def get_weather(get_builder):
    return get_builder.weather


def test_concrete_weather_builder(get_weather) -> None:
    assert isinstance(get_weather, WeatherData), 'not right class'
    assert len(get_weather.df.columns) == 24, 'length is not 24'


@pytest.mark.chk_dtypes
def test_concrete_weather_builder_change_dtypes(get_builder, get_weather) -> None:
    assert get_weather.df['temp'].dtype == np.float64, 'datetime is not a object'
    get_builder.change_dtypes('temp', np.float32)
    assert get_weather.df['temp'].dtype == np.float32, 'datetime is not a datetime'


@pytest.mark.chk_encoding
def test_concrete_weather_builder_multilabel_conditions(get_builder,
                                                        get_weather,
                                                        multilabel_conditions_columns) -> None:
    origin_columns_len = len(get_weather.df.columns)
    assert origin_columns_len == 24, 'not length of columns'
    get_builder.convert_multilabel_encoding('conditions')
    new_columns_len = len(get_weather.df.columns)
    assert new_columns_len > 24, 'not length of columns after encoding'
    assert new_columns_len - origin_columns_len == 4, 'not length of columns after encoding'
    assert 'conditions_Snow' in get_weather.df.columns, 'not equal arrays'


@pytest.mark.chk_encoding
def test_concrete_weather_builder_ohe_icon(get_builder,
                                           get_weather,
                                           multilabel_conditions_columns) -> None:
    assert get_weather.df.icon.dtype == 'object', 'its not object type'
    get_builder.convert_categorical_ohe(['icon'])
    assert 'ohe_clear-night' in get_weather.df.columns, 'clear-night not found in new columns'


@pytest.mark.add_feature
def test_add_season_feat(get_builder, get_weather) -> None:
    assert 'season' not in get_weather.df.columns, 'season column already exists'
    get_builder.add_season_feat()
    assert 'season' in get_weather.df.columns, 'season column doesnt exist'
    assert set(get_weather.df.season.unique()).issubset({0, 1, 2, 3}), 'not right value in seasons'


@pytest.mark.drop_columns
def test_drop_columns(get_builder) -> None:
    assert 'preciptype' in get_builder.weather.df.columns, 'preciptype not in columns'
    get_builder.drop_columns(['preciptype'])
    assert 'preciptype' not in get_builder.weather.df.columns, 'preciptype still in columns'

# director = Director(weather_data)
# builder = ConcreteBuilderWeather()
# director.builder = builder
# if __name__ == '__main__':
#     import os
#     print(os.getcwd())