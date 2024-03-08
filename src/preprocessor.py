from typing import Tuple, Dict, Any

import numpy
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series

import config
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)
from utils import console


def load_data(path: str, columns: list, sep=',', dec='.') -> pd.DataFrame:
    """
    :param dec: csv decimal symbol present in file
    :param sep: csv separator present in file
    :param path: path to the file
    :param columns: list containing the columns needed from the csv
    :return: a pd.Dataframe containing only the chosen columns
    """
    df = pd.read_csv(path, sep=sep, decimal=dec)
    df = df[columns]
    console.log(f"Model run with {columns}")
    num_na = df.isna().any(axis=1).sum()
    num_yes = len(df)
    df = df.dropna().reset_index()
    df = df.drop(columns='index')
    console.log(f"Deleted rows containing NAs ({num_na}/{num_yes} leaving {num_yes-num_na} rows)")

    return df


def compare_data_loader(base_path: str, columns: list, sep=',', dec='.'):
    """
    Function to read in data that was already previously split
    :param base_path: folder where to find the 4 needed .csv
    :param columns: columns to be used in the network
    :param sep:
    :param dec:
    :return: X_train, X_test, y_train, y_test
    """

    if 'd13C_cor' in columns:
        columns.remove('d13C_cor')
    console.log(f"Model run with {columns}")
    X_train = pd.read_csv(f'{base_path}/JB_X_train.csv', sep=sep, decimal=dec)
    X_train = X_train[columns]
    X_test = pd.read_csv(f'{base_path}/JB_X_test.csv', sep=sep, decimal=dec)
    X_test = X_test[columns]
    y_train = pd.read_csv(f'{base_path}/JB_y_train.csv', sep=sep, decimal=dec).values.ravel()
    y_test = pd.read_csv(f'{base_path}/JB_y_test.csv', sep=sep, decimal=dec).values.ravel()

    return X_train, X_test, y_train, y_test


def one_hot(df: pd.DataFrame, column: str) -> tuple[DataFrame, dict[Any, Any]]:
    """
    :param df: dataframe to be encoded
    :param column: a column containing strings to be encoded into columns
    :return: a pd.Dataframe with the original str-column removed and encoded, at the end
    """
    enc = OneHotEncoder()
    df_enc = enc.fit_transform(df[[column]])
    df_out = pd.DataFrame(df_enc.A, columns=enc.categories_[0])
    out = pd.concat([df.drop(columns=column), df_out], axis=1).reindex(df.index)
    one_hot_dict = {column: [enc.categories_[0]]}

    console.log(f"On-hot encoder for {column}: (rows are sorted) ")
    console.log(f"In this order: {enc.categories_[0]}")
    return out, one_hot_dict


def save(df: pd.DataFrame, out_path: str):
    df.to_csv(out_path, sep=',', index=False)


def date_to_year(data: pd.DataFrame, column: str):
    """
    Converts a date with the format dd.mm.yyyy to just the year, deletes the old date column
    :param data: dataframe
    :param column: column containing the sampling date in %d.%m.%Y to be changed to year
    :return: return a column year with that sample year, DELETES THE column !!
    """
    data['year'] = pd.to_datetime(data[column], format='%d.%m.%Y').dt.year
    data['year'] = pd.to_numeric(data['year'])
    data = data.drop(columns=column)
    return data


def scaling_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scales data using StandardScaler
    :param data: The dataframe to be scaled.
    :return: The scaled dataframe.
    """
    scaler = StandardScaler()
    names = data.columns
    scaled_data = scaler.fit_transform(data)
    scaled_out = pd.DataFrame(scaled_data, columns=names)
    console.log(f"Scaled data for columns:{data.columns}")
    return scaled_out


