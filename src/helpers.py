import pandas as pd
from glob import glob
import os


def load_df_from_dir(data_dir: str = '') -> pd.DataFrame:
    if not data_dir:
        return pd.DataFrame([])

    # Use glob to find all CSV files in the directory
    csv_files = glob(os.path.join(data_dir, '*.csv'))
    # Initialize an empty list to store DataFrames
    data_frames = []

    # Read each CSV file and append it to the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # dropping 'preciptype' column
        # df.drop(columns=['preciptype'], inplace=True)
        data_frames.append(df)

    # Horizontally concatenate the DataFrames
    combined_df = pd.concat(data_frames, axis=0, ignore_index=True)
    # Convert 'datetime' column to datetime type
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'], errors='coerce')
    # Sort by the 'datetime' column
    combined_df.sort_values(by='datetime', inplace=True)
    # Reset the index if needed
    combined_df.reset_index(drop=True, inplace=True)
    # combined_df.set_index('datetime', inplace=True)

    return combined_df


def build_print_line(txt: str, symbol: str = '-', repeat_times: int = 20) -> None:
    print(f'{symbol*repeat_times}{txt}{symbol*repeat_times}')


def missing_values(data: pd.DataFrame = None) -> None:
    build_print_line('check missing values')
    for col in data:
        missing_data = data[col].isna().sum()
        if missing_data > 0:
            perc = missing_data / len(data) * 100
            print(f'Feature {col} >> Missing entries: {missing_data} \
                |  Percentage: {round(perc, 2)} \
                |  Data Type: {data[col].dtypes}')

    build_print_line('check missing values end')
