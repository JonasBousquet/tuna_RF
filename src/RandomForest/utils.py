import os
import json
import pickle
import pandas as pd
from time import sleep
from datetime import datetime
from rich.console import Console
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score

console = Console(record=True)


def generate_run_directories(tag: str):
    """
    :param tag: The tag for the run directory.
    :return:the paths to the error directory, importance directory, main directory, validation curves directory,
     model directory, and plot directory.
    """
    main_dir = init_dir(root_dir="../runs", tag=tag)
    plot_dir = os.path.join(main_dir, "plots")
    importance_dir = os.path.join(plot_dir, "feature_importance")
    val_curves_dir = os.path.join(plot_dir, "validation_curves")
    res_curves_dir = os.path.join(plot_dir, "responses_curves")
    model_dir = os.path.join(main_dir, "models/")
    error_dir = os.path.join(plot_dir, "plots_error")
    os.mkdir(plot_dir)
    os.mkdir(importance_dir)
    os.mkdir(val_curves_dir)
    os.mkdir(res_curves_dir)
    os.mkdir(error_dir)
    os.mkdir(model_dir)
    console.log("[green]Starting the pipeline!")
    sleep(0.75)
    return error_dir, importance_dir, main_dir, val_curves_dir, model_dir, plot_dir


def init_dir(root_dir: str = "runs",
             tag: str = "") -> str:
    """
    :param root_dir: The root directory path.
    :param tag: The tag for the run directory
    :return: The run directory path
    """
    if not os.path.exists(root_dir):
        print(f"-> Creating root dir: {root_dir}")
        os.mkdir(root_dir)
    if tag != "":
        run_dir = os.path.join(root_dir,
                               str(datetime.now().
                                   strftime("pipeline_RF_%d.%m.%y_%Hh%M%S") + "_" + tag))
    else:
        run_dir = os.path.join(root_dir,
                               datetime.now().
                               strftime("pipeline_%d.%m.%y_%Hh%M%S"))
    if not os.path.exists(run_dir):
        print(f"-> Creating run directory: {run_dir}")
        os.mkdir(run_dir)
    return run_dir


def print_regression_metrics(y_true,
                             y_pred):
    """
    Regression metrics (R2, MAE, MSE, VAR)
    :param y_true: The true values.
    :param y_pred: The predicted values.
    :return:
    """
    sleep(0.75)
    console.log(f"[bold green]Regression metrics: \n"
                 f"    -> R2:  {r2_score(y_true=y_true, y_pred=y_pred)}\n"
                 f"    -> MAE: {mean_absolute_error(y_true=y_true, y_pred=y_pred)}\n"
                 f"    -> MSE: {mean_squared_error(y_true=y_true, y_pred=y_pred)}\n"
                 f"    -> VAR: {explained_variance_score(y_true=y_true, y_pred=y_pred)}\n")


def save_model(model,
               path="./model.pkl"):
    """
    Saves the model as a pickle file (.pkl)
    :param model: The model to be saved.
    :param path: (optional) The path to save the model. If not specified the name is model.pkl
    :return:
    """
    with open(path, mode="wb") as outfile:
        pickle.dump(model, outfile)


def save_params(logdir: str,
                filename: str,
                params: object):
    """
    Save the parameters.
    :param logdir: The directory to save the parameters.
    :param filename: The name of the file.
    :param params: The parameters to be saved.
    """
    with open(os.path.join(logdir, f"{filename}_parameters_.json"), 'w', encoding='utf8') as f:
        json.dump(params, f, indent=2)


def runtag2title(runtag: str):
    """
    Convert run tag to title. (may get removed)
    :param runtag: The run tag.
    :return: The converted title.
    """
    return runtag.replace('_', ' ')


def process_dataframe(df: pd.DataFrame,
                      encoder: dict):
    """
    Decode previously encoded columns using the dictionary created by encode_dict()
    :param df: The dataframe to be processed.
    :param encoder: The encoder dictionary.
    :return: The decoded dataframe.
    """
    for key, items in encoder.items():
        if rows_to_sum := list(items):
            rows_to_sum = list(rows_to_sum[0])
            df.loc[key] = df.loc[rows_to_sum].sum()
            df.drop(index=rows_to_sum, inplace=True)
        df.sort_values(by='importances', ascending=True, inplace=True)
    return df


def length_short(df: pd.DataFrame):
    """
    Rename dataframe index for beauty reasons
    :param df: The dataframe to be renamed.
    :return: The renamed dataframe.
    """
    return df.rename(index={'length_cm': 'length', 'c_ocean': 'ocean', 'c_sp_fao': 'species', 'sample_year': 'year'})


