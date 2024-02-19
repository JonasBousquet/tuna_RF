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
    :return:
    """
    df = pd.read_csv(path, sep=sep, decimal=dec)
    df = df[columns]
    return df



def one_hot(df: pd.DataFrame, column: str):

    enc = OneHotEncoder()
    unique_values = df[column].unique()
    df_enc = enc.fit_transform(df[[column]])
    df_out = pd.DataFrame(df_enc.A)


    out = pd.concat([df.drop(columns=column), df_out], axis=1).reindex(df.index)
    console.log(f"On-hot encoder for {column}: \n")
    for i in df[column].unique():
        console.log(f"* item: {[i]} -> num: {enc.transform([i])}")
    return out


def save(df: pd.DataFrame, out_path):
    df.to_csv(out_path, sep=',', index=False)
