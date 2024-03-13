import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def live_feature_importance(model, plot_dir: str, runtag: str, encoder=None):
    importances = model.feature_importances_
    colnames = model.feature_names_in_
    df = pd.DataFrame(importances, index=colnames, columns=['importances']).sort_values(by='importances')
    plot_df = utils.process_dataframe(df, encoder) if encoder is not None else df
    plot_df = utils.length_short(plot_df)
    # percent = plot_df['importances'] / plot_df['importances'].sum() * 100
    indices = range(len(plot_df.index))

    fig, ax = plt.subplots()
    ax.barh(indices, plot_df['importances'], align='center')
    # for i, (value, percent) in enumerate(zip(plot_df['importances'], percent)):
    #     ax.text(value + .01, i, f"{percent:.2f}%", fontsize=12, fontweight='bold', va='center')

    ax.set_yticks(indices, plot_df.index, weight='bold')
    ax.set_xlabel('Relative Importance')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    name = f"{plot_dir}/feature_importance/{runtag}_feature_importance.png"
    fig.savefig(name, bbox_inches='tight', transparent=True)
    print(f"Plot has been saved as {name}")
    fig.show()


def pred_vs_real(y_pred: pd.DataFrame, y_test: pd.DataFrame, plotdir: str, runtag: str, target: str):
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
    plt.scatter(y_test, y_pred, color='blue')  # data points
    if target == 'd13C_cor':
        limits = [-19, -13]
        ax_text = '$\\delta^{13}C$'
    elif target == 'logHg':
        limits = [-2, 1]
        ax_text = 'LogHg'
    else:
        print('Target not found, please add it to the code (pretty much everywhere)')
        return None
    plt.xlim(limits[0]+(limits[0]*0.05), limits[1]+(abs(limits[1])*0.05))
    plt.ylim(limits[0]+(limits[0]*0.05), limits[1]+(abs(limits[1])*0.05))
    plt.plot(range(limits[0]-1, limits[1]+2), range(limits[0]-1, limits[1]+2),
             color='red')  # red straight {1;1} line
    x_values, y_values, r2, slope, rmse = lin_reg(y_pred, y_test, limits)  # linear model
    plt.plot(x_values, y_values, linestyle='-.', color='black')  # Plotting the linear model
    plt.xlabel(f'measured {ax_text}')
    plt.ylabel(f'predicted {ax_text}')
    text = plt.annotate(f"n= {len(y_test)}\n$r^2$= {r2}\nSlope= {slope}\nRMSE= {rmse}",
                        xy=(0.04, 0.775),
                        xycoords='axes fraction',
                        weight='bold')
    text.set_bbox({'facecolor': 'white',
                   'edgecolor': 'black',
                   'linewidth': .5})
    name = f"{pred_dir}/{runtag}.png"
    plt.savefig(name,
                bbox_inches='tight',
                transparent=True)
    print(f"Plot as been saved as {name}")
    plt.show()


def lin_reg(pred: pd.DataFrame, origin: pd.DataFrame, limits: list):
    """
    Simple linear regression from predictions vs original data
    :param pred: predicted values
    :param origin: original values
    :return: x and y values for plotting and r2, slope, rmse to show on the plot
    """
    model = LinearRegression()
    pred = pred.reshape(-1, 1)
    if type(origin) is not np.array:
        origin = np.array(origin)
    origin = origin.reshape(-1, 1)
    res = model.fit(origin, pred)
    x_values = np.array(list(range(limits[0]-1, limits[1]+2))).reshape(-1, 1)
    y_values = res.predict(x_values)

    r2 = np.round(model.score(origin, pred), 4)
    slope = np.round(model.coef_[0], 4)[0]
    rmse = np.round(mean_squared_error(origin, pred, squared=False), 4)
    return x_values, y_values, r2, slope, rmse
