import config
import pickle
import pandas as pd
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
    return complete_dir, model_dir, plot_dir


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


def load_train_data(data_path, vars_in):

    X_test = pd.read_csv(f'{data_path}/JB_X_test.csv')
    y_test = pd.read_csv(f'{data_path}/JB_y_test.csv')
    X_test = X_test[vars_in]
    return X_test, y_test


def model_fit(model: object,
              X_test: pd.DataFrame,
              y_test: pd.DataFrame):

    y_pred = model.predict(X_test)

    metrics = get_regression_metrics(y_test, y_pred)


def get_regression_metrics(y_true, y_pred) -> dict:

    r2= r2_score(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    var = explained_variance_score(y_true=y_true, y_pred=y_pred)

    return {'r2': r2, 'mae': mae, 'mse': mse, 'var': var}
