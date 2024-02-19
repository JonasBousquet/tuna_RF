import json
import pickle
from time import sleep
from rich.console import Console
import os
from datetime import datetime
import matplotlib as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score

console = Console(record=True)


def generate_run_directories(tag: str):
    main_dir = init_dir(root_dir="./runs", tag=tag)
    plot_dir = os.path.join(main_dir, "plots")
    importance_dir = os.path.join(plot_dir, "feature_importance")
    val_curves_dir = os.path.join(plot_dir, "validation_curves")
    model_dir = os.path.join(main_dir, "models/")
    error_dir = os.path.join(plot_dir, "plots_error")
    os.mkdir(plot_dir)
    os.mkdir(importance_dir)
    os.mkdir(val_curves_dir)
    os.mkdir(error_dir)
    os.mkdir(model_dir)
    console.log(f"[green]Starting the pipeline!")
    sleep(0.75)
    return error_dir, importance_dir, main_dir, val_curves_dir, model_dir


def init_dir(root_dir: str = "runs", tag: str = "") -> str:
    if not os.path.exists(root_dir):
        print(f"-> Creating root dir: {root_dir}")
        os.mkdir(root_dir)
    if tag != "":
        run_dir = os.path.join(root_dir,
                               str(datetime.now().
                                   strftime("pipeline_%d_%m_%Y-%Hh%M") + "-" + tag))
    else:
        run_dir = os.path.join(root_dir,
                               datetime.now().
                               strftime("pipeline_%d_%m_%Y-%Hh%M"))
    if not os.path.exists(run_dir):
        print(f"-> Creating run directory: {run_dir}")
        os.mkdir(run_dir)
    return run_dir


def print_regression_metrics(y_true, y_pred):
    sleep(0.75)

    console.log(f"[bold green]Regression metrics: \n"
                 f"    -> R2:  {r2_score(y_true=y_true, y_pred=y_pred)}\n"
                 f"    -> MAE: {mean_absolute_error(y_true=y_true, y_pred=y_pred)}\n"
                 f"    -> MSE: {mean_squared_error(y_true=y_true, y_pred=y_pred)}\n"
                 f"    -> VAR: {explained_variance_score(y_true=y_true, y_pred=y_pred)}\n")


def save_model(model, path="./model.pkl"):
    with open(path, mode="wb") as outfile:
        pickle.dump(model, outfile)


def save_params(logdir, filename, params):
    with open(os.path.join(logdir, filename + "_parameters_.json"), 'w', encoding='utf8') as f:
        json.dump(params, f, indent=2)


def make_plot():

    return plot