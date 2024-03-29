import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.RandomForest import utils
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def pred_vs_real(y_pred: pd.DataFrame, y_train: pd.DataFrame, plotdir: str, runtag: str):
    """
    Plot predictions vs real data from the training set
    :param y_pred: the predicted values
    :param y_train: the actual values
    :param plotdir: main plot directory to save the plot
    :param runtag: name of the run, used to name the file and put correct title
    :return: Nothing, saves the plot
    """
    pred_dir = f'{plotdir}/validation_curves/'
    plt.rc('axes', axisbelow=True)
    plt.figure()
    plt.grid(linewidth=.25)
    plt.scatter(y_train, y_pred, color='blue')  # data points
    plt.plot(range(-19, -12), range(-19, -12), color='red')  # red straight {1;1} line
    x_values, y_values, r2, slope, rmse = lin_reg(y_pred, y_train)    # linear model
    plt.plot(x_values, y_values, linestyle='-.', color='black')  # Plotting the linear model
    plt.xlabel('measured $\\delta^{13}C$')
    plt.ylabel('predicted $\\delta^{13}C$')
    plt.xlim(-19.2, -12.8)
    plt.ylim(-19.2, -12.8)

    text = plt.annotate(f"n= {len(y_train)}\n$r^2$= {r2}\nSlope= {slope}\nRMSE= {rmse}", xy=(0.04, 0.775),
                        xycoords='axes fraction', weight='bold')
    text.set_bbox({'facecolor': 'white', 'edgecolor': 'black', 'linewidth': .5})
    name = f"{pred_dir}/{runtag}.png"
    plt.savefig(name, bbox_inches='tight', transparent=True)
    print(f"Plot as been saved as {name}")
    plt.show()


def lin_reg(pred: pd.DataFrame, origin: pd.DataFrame):
    """
    Simple linear regression from predictions vs original data
    :param pred: predicted values
    :param origin: original values
    :return: x and y values for plotting and r2, slope, rmse to show on the plot
    """
    model = LinearRegression()
    pred = pred.reshape(-1, 1)
    origin = origin.reshape(-1, 1)
    res = model.fit(origin, pred)
    x_values = np.array(list(range(-19, -12))).reshape(-1, 1)
    y_values = res.predict(x_values)

    r2 = np.round(model.score(origin, pred), 4)
    slope = np.round(model.coef_[0], 4)[0]
    rmse = np.round(mean_squared_error(origin, pred, squared=False), 4)
    return x_values, y_values, r2, slope, rmse
