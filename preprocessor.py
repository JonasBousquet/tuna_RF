import numpy
import pandas as pd
import config
from sklearn.preprocessing import OneHotEncoder
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
    return df


def one_hot(df: pd.DataFrame, column: str)-> pd.DataFrame:
    """
    :param df: dataframe to be encoded
    :param column: a column containing strings to be encoded into columns
    :return: a pd.Dataframe with the original str-column removed and encoded, at the end
    """
    enc = OneHotEncoder()
    unique_values = df[column].unique()
    df_enc = enc.fit_transform(df[[column]])
    df_out = pd.DataFrame(df_enc.A, columns=enc.categories_[0])

    out = pd.concat([df.drop(columns=column), df_out], axis=1).reindex(df.index)
    console.log(f"On-hot encoder for {column}: (rows are sorted) ")
    console.log(f"In this order: {enc.categories_[0]}")
    return out


def save(df: pd.DataFrame, out_path: str):
    df.to_csv(out_path, sep=',', index=False)


def date_to_year(data: pd.DataFrame, column: str):
    """
    :param data: dataframe
    :param column: column containing the sampling date in %d.%m.%Y to be changed to year
    :return: return a column year with that sample year, DELETES THE column !!
    """
    data['year'] = pd.to_datetime(data[column], format='%d.%m.%Y').dt.year
    data['year'] = pd.to_numeric(data['year'])
    data = data.drop(columns=column)
    return data

