import re
import shap
import config
import pickle
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (OneHotEncoder)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score


def paths(runname: str):
    """
    Create all necessary paths to model
    :param rundir: base directory of the model
    :return: complete_dir, model_dir, plot_dir
    """
    if config.saved:
        complete_dir = f'../saved_runs/{runname}'
    else:
        complete_dir = f'../runs/{runname}'
        print('You should save the model to saved_runs')

    model_dir = f'{complete_dir}/models/RandomForestRegressor.pkl'
    plot_dir = f'{complete_dir}/plots/'

    match = re.search(r'_(\d+[a-z]+\d+)_(\w+)$', runname)
    runtag = match.group(2) if match else None
    return complete_dir, model_dir, plot_dir, runtag


def load_model(pickle_path: str):
    """
    Load model saved as .pkl (move to model loader)
    :param pickle_path: Path to the model directory
    :return: fitted model
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    vars_in = data.feature_names_in_
    return data, vars_in


def load_data(data_path, vars_in):
    X_test = pd.read_csv(f'{data_path}/JB_X_test.csv')
    y_test = pd.read_csv(f'{data_path}/JB_y_test.csv')
    X_train = pd.read_csv(f'{data_path}/JB_X_train.csv')
    y_train = pd.read_csv(f'{data_path}/JB_y_train.csv')
    X_train_encoded, X_test_encoded, encode_dict = encode_data(X_train,
                                                               X_test)
    X_test = X_test_encoded[vars_in]
    X_train = X_train_encoded[vars_in]
    return X_test, y_test, X_train, y_train


def model_fit(model: RandomForestRegressor,
              X_test: pd.DataFrame,
              y_test: pd.DataFrame,
              X_train: pd.DataFrame,
              y_train: pd.DataFrame):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = get_regression_metrics(y_test, y_pred)
    return y_pred, metrics


def get_regression_metrics(y_true, y_pred) -> dict:
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    var = explained_variance_score(y_true=y_true, y_pred=y_pred)

    return {'r2': r2, 'mae': mae, 'mse': mse, 'var': var}


def shap_values(model: RandomForestRegressor,
                X_test: pd.DataFrame,
                runtag: str,
                plot_dir: str):
    timeformat = "%H:%M:%S"
    now = datetime.datetime.now()
    print(now.strftime(f'[{timeformat}] Fitting shap values or something'))
    # shap.initjs()
    # explainer = shap.KernelExplainer(model, X_train)
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f'{plot_dir}/feature_importance/{runtag}_shap_values.png')
    plt.show()
    now2 = datetime.datetime.now()
    runtime = now2 - now
    print(now2.strftime(f'[{timeformat}] Yeah it\'s done, after {round(runtime.total_seconds(), 2)} seconds'))


def encode_data(train_data: pd.DataFrame,
                test_data: pd.DataFrame):
    """
    Encodes the columns 'c_sp_fao' and 'c_ocean' if they are present
    :param test_data: X_test
    :param train_data: X_train
    :return: train_data, test_data with the encoded columns if present and the matiching encoder_dict
    """
    if 'c_sp_fao' in train_data.columns:
        train_data, dict1 = one_hot(train_data, 'c_sp_fao')
        test_data, _dict11 = one_hot(test_data, 'c_sp_fao')
    else:
        dict1 = None
    if 'c_ocean' in train_data.columns:
        train_data, dict2 = one_hot(train_data, 'c_ocean')
        test_data, _dict2 = one_hot(test_data, 'c_ocean')
    else:
        dict2 = None

    encoder_dict = encode_dict(dict1, dict2)

    return train_data, test_data, encoder_dict


def one_hot(df: pd.DataFrame,
            column: str):
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
    return out, one_hot_dict


def encode_dict(dict1: dict,
                dict2: dict):
    if dict1 and dict2 is not None:
        return merge_dict(dict1, dict2)
    elif dict1 is not None:
        return dict1
    elif dict2 is not None:
        return dict2
    else:
        return None


def merge_dict(dict1: dict,
               dict2: dict):
    out = dict1.copy()
    for key, value in dict2.items():
        out[key] = value
    return out


