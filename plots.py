import matplotlib.pyplot as plt
import config
import pickle
import re


def pickle_data(pickle_path: str):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_feature_importance(rundir: str, filename: str):
    # Paths and stuff
    rundir_com = 'runs/'+rundir
    model = pickle_data(rundir_com+'/models/RandomForestRegressor.pkl')
    fig_path = rundir_com+'/plots'

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
    plt.savefig(fig_path+'/feature_importance/'+filename)
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

