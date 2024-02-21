import matplotlib.pyplot as plt
import config
import pickle

def pickle_data(pickle_path: str):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_feature_importance(rundir: str, filename: str):
    rundir = 'runs/'+rundir
    model = pickle_data(rundir+'/models/RandomForestRegressor.pkl')
    fig_path = rundir+'/plots'

    importances = model.feature_importances_
    colnames = model.feature_names_in_

    indices = range(len(importances))
    names = [colnames[i] for i in importances.argsort()]
    plt.figure()
    plt.title('Feature Importance with Random Forest Regressor')
    plt.barh(indices, sorted(importances), align='center')
    plt.yticks(indices, names)
    plt.xlabel('Relative Importance')
    plt.savefig(fig_path+'/feature_importance/'+filename)
    plt.show()
