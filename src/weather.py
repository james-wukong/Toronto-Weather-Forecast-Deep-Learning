import logging
from collections import defaultdict
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='weather.log', filemode='w')


class WeatherData:

    def __init__(self, df: pd.DataFrame = None):
        self.df = df

    def get_feature_index(self, feat_list=None) -> list:
        feature_indices = [
            self.df.columns.get_loc(feature) for feature in feat_list
        ]

        return feature_indices

    @staticmethod
    def display_corr(
        df: pd.DataFrame = None,
        win_size: tuple[float, float] = (20, 16)) -> None:
        """
        display the corr of dataset
        Args:
            df: DataFrame
            win_size: tuple, display size of window
        Returns:

        """
        plt.figure(figsize=win_size)
        # plotting correlation heatmap
        sns.heatmap(df.corr(), cmap="coolwarm", annot=True)

        # displaying heatmap
        plt.show()

    @staticmethod
    def display_losses(history) -> None:
        """
        Plot the training and validation loss
        """
        # Extract training and validation metrics from the history object
        print(history.history)
        training_loss = history.history['loss']
        validation_loss = history.history.get('val_loss', None)
        training_accuracy = history.history.get('cls_out_accuracy', None)
        validation_accuracy = history.history.get('val_cls_out_accuracy', None)

        # Plotting the training loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(training_loss, label='Training Loss')
        if validation_loss is not None:
            plt.plot(validation_loss, label='Validation Loss')
            plt.legend()
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        # Plotting the training accuracy
        if training_accuracy is not None:
            plt.subplot(1, 2, 2)
            plt.plot(training_accuracy, label='Training Accuracy')
            if validation_accuracy is not None:
                plt.plot(validation_accuracy, label='Validation Accuracy')
                plt.legend()
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig('data/images/history.png')
        plt.show()

    @staticmethod
    def plot_test_predictions(y_test, y_test_hat, title='Test Predictions'):
        """
        Plot the test prediction
        """
        plt.plot(y_test, label='True Values', marker='.')
        plt.plot(y_test_hat, label='Predicted Values', marker='.')
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Original Values')  # Adjust based on your data scaling
        plt.legend()
        plt.show()

    @staticmethod
    def display_boxplot(df: pd.DataFrame, x: str, y: str) -> None:
        """
        display the boxplot to detect outliers
        Args:
            df: DataFrame, dataset
            x: str, x coordinate feature
            y: str, y coordinate feature

        Returns:

        """
        sns.boxplot(data=df, x=x, y=y, fill=False, gap=.1)

    @staticmethod
    def display_confusion_matrix(y_test,
                                 y_test_hat,
                                 title='Evaluation Confusion Matrix') -> None:
        """
        Plot the confusion matrix from test evaluation
        :param y_test: true value of y_test
        :param y_test_hat: predicted probability of y_test
        :param title:
        :return:
        """
        conf_matrix = confusion_matrix(y_test, y_test_hat)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=True,
                    yticklabels=True)
        plt.title(title)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()

    @staticmethod
    def display_roc_curve(y_test,
                          y_test_hat,
                          title='ROC Curve') -> None:
        fpr, tpr, _ = metrics.roc_curve(y_test, y_test_hat)
        auc = metrics.roc_auc_score(y_test, y_test_hat)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.title(title)
        plt.legend(loc=4)
        plt.show()


class BaseBuilder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @property
    @abstractmethod
    def weather(self) -> None:
        pass

    @weather.setter
    def weather(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def change_dtypes(self, col_name: str, to_col_dtype: str) -> None:
        pass

    @abstractmethod
    def remove_duplicated(self) -> None:
        pass

    @abstractmethod
    def drop_columns(self, cols=None) -> None:
        pass

    @abstractmethod
    def impute_missing_values(self,
                              col: str,
                              limit_direction='forward') -> None:
        pass

    @abstractmethod
    def impute_missing_values_single(self, col: str, value=0) -> None:
        pass

    @abstractmethod
    def convert_multilabel_encoding(self, col: str) -> None:
        pass

    @abstractmethod
    def convert_categorical_ohe(self, cols=None) -> None:
        pass

    @abstractmethod
    def convert_categorical_le(self, cols=None) -> None:
        pass

    @abstractmethod
    def convert_numerical_encoding(self,
                                   enc,
                                   cols=None,
                                   fit: bool = False) -> None:
        pass

    @abstractmethod
    def convert_cat2num_infer(self, enc, cols=None) -> None:
        pass

    @abstractmethod
    def check_missing_values(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def remove_col_white_space_l(self, col: str) -> None:
        pass

    @abstractmethod
    def remove_col_white_space_r(self, col: str) -> None:
        pass

    @abstractmethod
    def remove_col_white_space_all(self, col: str) -> None:
        pass

    @abstractmethod
    def convert_to_datetime(self, col: str) -> None:
        pass

    @abstractmethod
    def add_year_month_day_feats(self, col: str) -> None:
        pass

    @abstractmethod
    def add_hour_feat(self, col: str) -> None:
        pass

    @abstractmethod
    def add_day_of_week_feat(self, col: str) -> None:
        pass

    @abstractmethod
    def add_week_of_year_feat(self, col: str) -> None:
        pass

    @abstractmethod
    def convert_string_to_set(self, col: str) -> None:
        pass

    @abstractmethod
    def add_season_feat(self, col: str) -> None:
        pass

    @abstractmethod
    def sort_data_by(self, col: str, sort: bool = True) -> None:
        pass

    @abstractmethod
    def split_sequence(
            self,
            data: pd.DataFrame,
            input_win_size: int,
            output_win_size: int,
            label_columns: defaultdict = None) -> tuple[np.array, np.array]:
        pass

    @abstractmethod
    def inverse_label_sequence(self,
                               data: tuple,
                               label_columns: tuple = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def create_train_test(self,
                          input_win_size: int,
                          output_win_size: int,
                          train_size: float,
                          validate_size: float,
                          batch_size: int,
                          label_columns: defaultdict = None) -> tuple:
        pass

    def unscale_prediction(self,
                           input_win_size: int,
                           predictions: pd.DataFrame,
                           label_columns: defaultdict = None) -> np.ndarray:
        pass


class ConcreteBuilderWeather(BaseBuilder, ABC):
    """
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps. Your program may have
    several variations of Builders, implemented differently.
    """

    def __init__(self, df: pd.DataFrame = None) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        self._weather = WeatherData(df)
        # self.reset()
        self.scaler = MinMaxScaler()
        self.y_test_ds = None
        self.y_test_reg_ds = None
        self.y_test_cls_ds = None
        self.dataset = None
        self.predict_input = None
        self.predict_input_original = None
        self.selected_feats = [
            'temp', 'feelslike', 'dew', 'snowdepth', 'windgust', 'humidity',
            'precip', 'precipprob', 'snow', 'windspeed', 'winddir',
            'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
            'severerisk', 'day', 'month', 'year', 'dayofweek', 'weekofyear',
            'hour', 'season'
        ]

    # def reset(self) -> None:
    #     self._weather = WeatherData(self._df)

    @property
    def weather(self) -> WeatherData:
        """
        Concrete Builders are supposed to provide their own methods for
        retrieving results. That's because various types of builders may create
        entirely different products that don't follow the same interface.
        Therefore, such methods cannot be declared in the base Builder interface
        (at least in a statically typed programming language).
        """
        return self._weather

    @weather.setter
    def weather(self, df: pd.DataFrame) -> None:
        self._weather = df

    def change_dtypes(self, col_name: str, to_col_dtype: str) -> None:
        """
        Change column dtype to target dtype
        Args:
            col_name: str, col name that its dtype will be changed
            to_col_dtype: str, target dtype
        Return:
            df: pd.DataFrame
        """
        self.weather.df[col_name] = self.weather.df[col_name].astype(
            to_col_dtype)

    def remove_duplicated(self) -> None:
        """
        remove duplicated rows
        """
        if self.weather.df.duplicated().sum() > 0:
            self.weather.df.drop_duplicates(inplace=True)

    def drop_columns(self, cols=None) -> None:
        """
        drop columns that we don't need
        Args:
            cols: list, columns that need be dropped

        Returns:

        """
        if cols and isinstance(cols, list):
            self.weather.df = self.weather.df.drop(columns=cols)

    def impute_missing_values(self,
                              col: str,
                              limit_direction='forward') -> None:
        """
        fill in missing values with interpolate
        Args:
            limit_direction: str, 'forward' or 'backward'
            col: str, column need to apply to interpolate filling

        Returns:

        """
        self.weather.df[col].interpolate(method='linear',
                                         limit_direction=limit_direction,
                                         inplace=True)

    def impute_missing_values_single(self, col: str, value=0) -> None:
        """
        fill in missing values with single value
        Args:
            col: str, column name need to apply to
            value: value to be filling

        Returns:

        """
        self.weather.df[col].fillna(value=value, inplace=True)

    def convert_multilabel_encoding(self, col: str) -> None:
        """
        convert categorical column into numerical column
        Args:
            col: str, encoding column name
        """
        mlb = MultiLabelBinarizer()
        self.convert_string_to_set(col)
        mlb_encoded = pd.DataFrame(
            mlb.fit_transform(self.weather.df[col]),
            columns=col + '_' + mlb.classes_,
        )
        self.weather.df = self.weather.df.reset_index(drop=True)
        mlb_encoded = mlb_encoded.reset_index(drop=True)

        self.weather.df = pd.concat(
            [mlb_encoded,
             self.weather.df.drop(columns=[col], axis=1)], axis=1)

    def convert_categorical_ohe(self, cols=None) -> None:
        """
        categorical one hot encoding of columns
        Args:
            cols: list, columns applying one hot encoding

        Returns:

        """
        if not cols or not isinstance(cols, list):
            return None
        # outputs array instead of dataframe
        ohe = OneHotEncoder(handle_unknown='ignore')
        array_hot_encoded = ohe.fit_transform(self.weather.df[cols]).toarray()
        feature_labels = ohe.categories_
        feature_labels = 'ohe_' + np.array(feature_labels).ravel()
        data_hot_encoded = pd.DataFrame(array_hot_encoded,
                                        columns=feature_labels)

        self.weather.df = self.weather.df.reset_index(drop=True)
        data_hot_encoded = data_hot_encoded.reset_index(drop=True)

        self.weather.df = pd.concat(
            [data_hot_encoded,
             self.weather.df.drop(cols, axis=1)], axis=1)

    def convert_categorical_le(self, cols=None) -> None:
        """
        categorical label encoding of columns
        Args:
            cols: list, columns applying label encoding

        Returns:

        """
        le = LabelEncoder()
        self.weather.df[cols] = self.weather.df[cols].apply(
            lambda col: le.fit_transform(col))

    def convert_cat2num_infer(self, enc, cols=None) -> None:
        """
        convert categorical column into numerical column
        Args:
            enc: type of encoding used to convert
            cols: list of columns need to be converted
        """
        pass

    def convert_numerical_encoding(self,
                                   enc,
                                   cols=None,
                                   fit: bool = False) -> None:
        """
        convert numerical columns into normalization or standardization form
        Args:
            fit: if this encoding requires fit_transform, fit = True else fit = False
            enc: function, encoding method (MinMaxScaler, StandardScaler)
            cols: list, columns get encoded

        Returns:

        """
        if not cols:
            return None
        if fit:
            self.weather.df[cols] = enc.fit_transform(self.weather.df[cols])
        else:
            self.weather.df[cols] = enc.transform(self.weather.df[cols])

    def check_missing_values(self) -> pd.DataFrame:
        """

        Returns:
            df: DataFrame with missing values
        """
        return (self.weather.df.isnull().sum().sort_values(ascending=False))

    def remove_col_white_space_l(self, col: str) -> None:
        """
        remove white space at the beginning of string
        Args:
            col: str, column name

        Returns:

        """
        self.weather.df[col] = self.weather.df[col].str.lstrip()

    def remove_col_white_space_r(self, col: str) -> None:
        """
        remove white space at the end of string
        Args:
            col: str, column name

        Returns:

        """
        self.weather.df[col] = self.weather.df[col].str.strip()

    def remove_col_white_space_all(self, col: str) -> None:
        """
        remove all white space
        Args:
            col: str, column name

        Returns:

        """
        self.weather.df[col] = self.weather.df[col].apply(
            lambda data: str(data).replace(' ', ''))

    def convert_to_datetime(self, col: str = 'datetime') -> None:
        """
        Convert column to pandas datetime for further processing
        Args:
            col: str, date column in object or string format

        Returns:

        """
        if not pd.api.types.is_datetime64_dtype(self.weather.df.datetime):
            self.weather.df[col] = pd.to_datetime(self.weather.df[col],
                                                  errors='coerce')

    def add_year_month_day_feats(self, col: str = 'datetime') -> None:
        """
        Add year, month, and day features to dataset
        Args:
            col: str, create day features based on this column

        Returns:

        """
        if not pd.api.types.is_datetime64_dtype(self.weather.df.datetime):
            self.convert_to_datetime(col)
        self.weather.df['day'] = self.weather.df[col].dt.day
        self.weather.df['month'] = self.weather.df[col].dt.month
        self.weather.df['year'] = self.weather.df[col].dt.year

    def add_hour_feat(self, col: str = 'datetime') -> None:
        """
        Add hour feature to dataset
        Args:
            col: str, create day features based on this column

        Returns:

        """
        self.weather.df['hour'] = self.weather.df[col].dt.hour

    def add_day_of_week_feat(self, col: str = 'datetime') -> None:
        """
        Add a feature column: day of week
        Args:
            col: str, create day of week based on this column
        Returns:

        """
        self.weather.df['dayofweek'] = self.weather.df[col].dt.dayofweek

    def add_week_of_year_feat(self, col: str = 'datetime') -> None:
        """
        Add week of year feature to dataset
        Args:
            col: str, create week of year based on this column

        Returns:

        """

        self.weather.df['weekofyear'] = self.weather.df[col].apply(
            lambda date: date.isocalendar().week if pd.notna(date) else pd.NaT)

    def convert_string_to_set(self, col: str) -> None:
        """
        Convert str column into set by splitting ','
        Args:
            col: str, column name to be proceeded

        Returns:

        """
        self.weather.df[col] = self.weather.df[col].apply(
            lambda data: list(set(str(data).replace(' ', '').split(','))))

    def add_season_feat(self, col: str = 'datetime') -> None:
        """
        Add season feature into dataframe
        Args:
            col: str, column name. create season column based on this column

        Returns:

        """
        # Toronto Seasons format from (month, day) to (month, day)
        spring_s, spring_e = (3, 20), (6, 20)
        summer_s, summer_e = (6, 21), (9, 21)
        fall_s, fall_e = (9, 22), (12, 20)

        # winter_s, winter_e = (12, 21), (3, 19)

        def get_season(dt) -> int:
            # spring = 0, summer = 1, fall = 2, winter = 3
            season, month, day = 0, dt.month, dt.day
            curr_day = (month, day)
            if spring_e <= curr_day <= spring_e:
                season = 0
            elif summer_s <= curr_day <= summer_e:
                season = 1
            elif fall_s <= curr_day <= fall_e:
                season = 2
            else:
                season = 3

            return season

        if not pd.api.types.is_datetime64_dtype(self.weather.df.datetime):
            self.convert_to_datetime(col)
        self.weather.df['season'] = self.weather.df[col].apply(get_season)

    def sort_data_by(self, col: str, sort: bool = True) -> None:
        """
        after completion of data clean and preprocessing, sort by col
        Args:
            col: str, columns name
            sort: str, 'ascending' or 'descending'

        Returns:

        """
        self.weather.df.sort_values(by=col, ascending=sort, inplace=True)

    def split_sequence(
        self,
        data: pd.DataFrame,
        input_win_size: int,
        output_win_size: int,
        label_columns: defaultdict = None
    ) -> tuple[np.array, np.array, np.array]:
        """
        split data into feature sequences and label sequences with time steps
        Args:
            data: DataFrame, input data
            label_columns: tuple, contain features to be list
            input_win_size: # 5 days of hourly observations -> 5 * 24
            output_win_size:  # Predict weather for the next 3 days -> 3 * 24

        Returns:
            tuple(sequences, regression labels, classification labels)
        """
        sequences, labels_reg, labels_cls = [], [], []
        for i in range(len(data) - input_win_size - output_win_size + 1):
            seq = data.iloc[i:i + input_win_size, :].values
            if label_columns is not None:
                label_reg = (data.iloc[
                    i + input_win_size:i + input_win_size + output_win_size,
                    data.columns.get_indexer(label_columns['reg'])].values)
                label_cls = (data.iloc[
                    i + input_win_size:i + input_win_size + output_win_size,
                    data.columns.get_indexer(label_columns['cls'])].values)
                labels_reg.append(label_reg)
                labels_cls.append(label_cls)
            sequences.append(seq)

        return (np.array(sequences, dtype=np.float32),
                np.array(labels_reg, dtype=np.float32),
                np.array(labels_cls, dtype=np.float32))

    def inverse_label_sequence(self,
                               y_data_hat: np.array,
                               label_columns: tuple = None) -> pd.DataFrame:
        """
        inverse split sequence and generate an original dataset
        Args:
            y_data_hat: np.array, input data
            label_columns: tuple, contain features to be list

        Returns:
        """
        result = pd.DataFrame()
        for idx, labels in enumerate(y_data_hat):
            if idx == 0:
                a = pd.DataFrame(labels[:, :], columns=label_columns)
                # result.append(labels[:, :], columns=label_columns, ignore_index=True, inplace=True)
            else:
                a = pd.DataFrame([labels[-1][:]], columns=label_columns)
                # result.append([labels[-1][:]], columns=label_columns)
            result = pd.concat([result, a], ignore_index=True)

        return result

    def create_train_test(self,
                          input_win_size: int,
                          output_win_size: int,
                          train_size: float = 0.8,
                          validate_size: float = 0.1,
                          batch_size: int = 24 * 30,
                          label_columns: defaultdict = None) -> tuple:
        """
        create train test dataset
        Args:
            label_columns: dict {'regression': (), 'classification': ()}
            input_win_size: int, n_step_in
            output_win_size: int, n_step_out
            train_size: float, training fraction of dataset
            validate_size: float, validation fraction of dataset
            batch_size: int, batch size of training dataset

        Returns:

        """
        train_size = int(train_size * len(self.weather.df))
        val_size = int(validate_size * len(self.weather.df))

        # 1. split dataset into training, validating, and test datasets
        train_data = self.weather.df.iloc[:train_size]
        val_data = self.weather.df.iloc[train_size:train_size + val_size]
        test_data = self.weather.df.iloc[train_size + val_size:]
        self.predict_input_original = self.weather.df.iloc[
            -1 * input_win_size:,
            self.weather.df.columns.get_indexer(label_columns['reg'] +
                                                label_columns['cls'])]
        # test_data.to_csv('test_data.csv', columns=['temp', 'feelslike', 'year', 'month', 'day', 'hour', 'season'])

        selected_feats_indices = self.weather.df.columns.get_indexer(
            self.selected_feats)
        # self.weather.df.iloc[-30:, selected_feats_indices].to_csv('test_data.csv', sep=',')
        # 2. scaling numerical features for all datasets
        train_data.iloc[:, selected_feats_indices] = self.scaler.fit_transform(
            train_data[self.selected_feats])
        val_data.iloc[:, selected_feats_indices] = self.scaler.transform(
            val_data[self.selected_feats])
        test_data.iloc[:, selected_feats_indices] = self.scaler.transform(
            test_data[self.selected_feats])
        self.dataset = pd.concat([train_data, val_data, test_data],
                                 ignore_index=True)
        # self.dataset.iloc[-30:, selected_feats_indices].to_csv('scaled data.csv', sep=',')
        self.predict_input, _, _ = self.split_sequence(
            self.dataset.iloc[-1 * input_win_size:],
            input_win_size,
            output_win_size=0,
            label_columns=None)

        # print('train_data shape: ', train_data.shape)
        # print('val_data: ', val_data.shape)
        # print('test_data: ', test_data.shape)

        # logging debug info
        # logging.info(f'X_train data after seq: {train_data} X_train shape {train_data.shape}')
        # train_data.to_csv('X_train.csv', sep=',')

        # 3. split datasets into sequences wrt n_step_in, and n_step_out
        X_train, y_train_reg, y_train_cls = self.split_sequence(
            train_data, input_win_size, output_win_size, label_columns)
        X_val, y_val_reg, y_val_cls = self.split_sequence(
            val_data, input_win_size, output_win_size, label_columns)
        X_test, y_test_reg, y_test_cls = self.split_sequence(
            test_data, input_win_size, output_win_size, label_columns)
        self.y_test_reg_ds, self.y_test_cls_ds = y_test_reg.copy(
        ), y_test_cls.copy()
        # np.savetxt('X_test.csv', X_test, delimiter=",")
        # print('X_train: ', X_train.shape, 'y_train_reg: ', y_train_reg.shape)
        # print('X_val: ', X_val.shape, 'y_val_reg: ', y_val_reg.shape)
        # print('X_test: ', X_test.shape, 'y_test_reg: ', y_test_reg.shape)

        # 4. batch and shuffle datasets and load them into tensors
        train_dataset = (tf.data.Dataset.from_tensor_slices(
            (X_train, y_train_reg,
             y_train_cls)).batch(batch_size).shuffle(len(X_train)))
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val_reg, y_val_cls)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (X_test, y_test_reg, y_test_cls)).batch(batch_size)

        # return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        return train_dataset, val_dataset, test_dataset

    def unscale_prediction(self,
                           input_win_size: int,
                           predictions: pd.DataFrame,
                           label_columns: defaultdict = None) -> np.ndarray:
        result = np.ndarray
        selected_label_indices = [
            self.selected_feats.index(label) for label in label_columns['reg']
        ]
        if not self.dataset.empty:
            empty_ds = np.empty((len(predictions), len(self.selected_feats)))
            empty_ds[:, :] = np.nan
            empty_ds[:len(predictions), selected_label_indices] = predictions

            result = self.scaler.inverse_transform(empty_ds)
            # np.savetxt('prediction inverse result.csv', result, delimiter=',')

        return result[:, selected_label_indices]


class Director:
    """
    The Director is only responsible for executing the building steps in a
    particular sequence. It is helpful when producing products according to a
    specific order or configuration. Strictly speaking, the Director class is
    optional, since the client can control builders directly.
    """

    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> BaseBuilder:
        return self._builder

    @builder.setter
    def builder(self, builder: BaseBuilder) -> None:
        """
        The Director works with any builder instance that the client code passes
        to it. This way, the client code may alter the lstm type of the newly
        assembled product.
        """
        self._builder = builder

    def build_weather_dataset(self) -> None:
        # 1. remove duplicated
        self.builder.remove_duplicated()
        # 2. handling missing data in preciptype
        # because it is quite similar with conditions
        # just drop this column, 'stations' and 'name' column
        self.builder.drop_columns(['preciptype', 'name'])
        self.builder.impute_missing_values('snow')
        self.builder.impute_missing_values('snowdepth')
        self.builder.impute_missing_values('sealevelpressure')
        self.builder.impute_missing_values_single('severerisk')
        self.builder.impute_missing_values_single(
            'visibility', value=self.builder.weather.df['visibility'].mean())
        self.builder.impute_missing_values_single(
            'windgust', value=self.builder.weather.df['windgust'].mean())
        # 3. change datetime column dtypes, sort the data by date
        self.builder.convert_to_datetime('datetime')
        # 4. Convert categorical features to numerical
        self.builder.convert_multilabel_encoding('conditions')
        self.builder.convert_multilabel_encoding('stations')
        self.builder.convert_categorical_ohe(['icon'])
        # 5. create time series features based on datetime
        self.builder.add_year_month_day_feats('datetime')
        self.builder.add_day_of_week_feat('datetime')
        self.builder.add_week_of_year_feat('datetime')
        self.builder.add_hour_feat('datetime')
        self.builder.add_season_feat('datetime')

        # sort data
        self.builder.sort_data_by('datetime')
        # 6. drop columns after correlation heatmap comparison
        self.builder.drop_columns(['solarenergy', 'uvindex', 'datetime'])
