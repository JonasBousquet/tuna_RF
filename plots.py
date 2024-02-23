import matplotlib.pyplot as plt
import config
import pickle
import re
import pandas as pd
import preprocessor as pre


def pickle_data(pickle_path: str):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def pathz(rundir: str):
    complete_dir = 'runs/'+ rundir
    model_dir = complete_dir + '/models/RandomForestRegressor.pkl'
    plot_dir = complete_dir + '/plots'
    return complete_dir, model_dir, plot_dir

def plot_feature_importance(rundir: str, filename: str):
    # Paths and stuff
    complete_dir, model_dir, plot_dir = pathz(rundir)
    model = pickle_data(model_dir)

    # get model information
    importances = model.feature_importances_
    colnames = model.feature_names_in_
    indices = range(len(importances))
    names = [colnames[i] for i in importances.argsort()]
    variance_ex = float(get_var_from_log(rundir))*100
    var_percentage_str = "{:.2f} %".format(variance_ex)

    # Plot the fig
    plt.figure()
    plt.title(f"Feature Importance with Random Forest Regressor \n Variance explained: {var_percentage_str}")
    plt.barh(indices, sorted(importances), align='center')
    plt.yticks(indices, names)
    plt.xlabel('Relative Importance')
    plt.savefig(plot_dir + '/feature_importance/' + filename)
    plt.show()


def get_var_from_log(rundir: str):
    log = 'runs/'+rundir+'/run_log.txt'

    with open(log, "r") as file:
        log_content = file.read()
    pattern = r"->\s*VAR:\s*([\d.]+)"
    matches = re.findall(pattern, log_content)
    if matches:
        var_value = matches[-1]
        return var_value
    else:
        print("VAR value not found in the log file.")


def data_vs_pred(data_dir: str, rundir: str, var_x: str, var_y: str):
    cols = [var_x, var_y]
    df = pre.load_data(data_dir, cols)
    complete_dir, model_dir, plot_dir = pathz(rundir)
    x_axis = df[var_x]
    y_axis = df[var_y]

    # Base fig with data
    plt.figure()
    plt.title(f"{var_x} vs {var_y} response")
    plt.scatter(x_axis, y_axis, color='grey', alpha=0.1)
    plt.xlabel(f"{var_x}")
    plt.ylabel(f"{var_y}")


    plt.show()

def predict_graph(rundir: str, data: str, var_x: str, var_y: str):

    complete_dir, model_dir, plot_dir = pathz(rundir)
    model = pickle_data(model_dir)
    pred = model.predict()


