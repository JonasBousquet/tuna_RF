import matplotlib.pyplot as plt
import config
import pickle
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import preprocessor as pre


def pickle_data(pickle_path: str):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def pathz(rundir: str):
    complete_dir = 'runs/'+ rundir
    model_dir = complete_dir + '/models/RandomForestRegressor.pkl'
    plot_dir = complete_dir + '/plots/'
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
    plt.savefig(plot_dir + 'feature_importance/' + filename)
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


def live_feature_importance(model, plot_dir: str, title):

    importances = model.feature_importances_
    colnames = model.feature_names_in_
    indices = range(len(importances))
    names = [colnames[i] for i in importances.argsort()]
    plt.figure()
    plt.title(f"Feature Importance \n Run: {title}")
    plt.barh(indices, sorted(importances), align='center')
    plt.yticks(indices, names)
    plt.xlabel('Relative Importance')
    name=plot_dir + '/feature_importance/plot.png'
    plt.savefig(name)
    print(f"Plot as been saved as {name}")
    #plt.show()


def pred_vs_real(y_pred: pd.DataFrame, y_train: pd.DataFrame, plotdir: str, title: str):

    plt.figure()
    plt.title(f"Actual $\\delta^{13}C$ vs predicted $\\delta^{13}C$ \n Run: {title}")
    # data points
    plt.scatter(y_train, y_pred, color='blue')
    # red straight {1;1} line
    plt.plot(range(-19, -13), range(-19, -13), color='red')
    # linear model
    x_values, y_values, r2, slope, rmse = lin_reg(y_pred, y_train)
    plt.plot(x_values, y_values, linestyle='-.', color='black')
    # Text & shit
    plt.xlabel('measured $\\delta^{13}C$')
    plt.ylabel('predicted $\\delta^{13}C$')
    plt.annotate(f"n= {len(y_train)}\n$R^2$= {r2}\nSlope= {slope}\nRMSE= {rmse}", xy=(0.04, 0.8), xycoords='axes fraction')

    name = plotdir+'/real_vs_pred.png'
    plt.savefig(name)
    print(f"Plot as been saved as {name}")
    plt.show()


def lin_reg(pred: pd.DataFrame, origin: pd.DataFrame):
    model = LinearRegression()
    pred = pred.reshape(-1, 1)
    origin = origin.reshape(-1, 1)
    res = model.fit(origin, pred)
    x_values = np.array([i for i in range(-19, -12)]).reshape(-1, 1)
    y_values = res.predict(x_values)

    r2 = np.round(model.score(origin, pred), 4)
    slope = np.round(model.coef_[0], 4)[0]
    rmse = np.round(root_mean_squared_error(origin, pred), 4)
    return x_values, y_values, r2, slope, rmse
