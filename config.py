from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Path
path_to_file = './data'
run_tag = 'plot_test'


# Regressor type
RFregressor = RandomForestRegressor()
random_forest_params = {"n_estimators": [500, 1000, 1500],
                        "min_samples_split": [3, 4, 5, 10],
                        "min_samples_leaf": [3, 4, 5, 10]
                        }
mlp_regressor = MLPRegressor()
mlp_param_grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive']
}



# Parameters
first_params = ["c_sp_fao", "SST", "d13C_cor", 'c_ocean']

reduced_params = ["sample_year", "d13Cdic", "SST", "d13C_cor"]

use_params = ["d13C_cor", "c_sp_fao", "c_ocean", "length_cm", "sample_year",
              "SST", "MLD", "Chl.a", "NPP", "d20", "d18", "d12", "O2_375m", "d13Cdic",
              "d13Cpom", "d15Npom", "doxycline", "dnethetero"]

coloring_vars = ["region", "region.col", "lon_dec", "lat_dec", "sample_owner", "d13C_pm"]


