from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

path_to_file = './data/db_d13c_sorted_utf.csv'

run_tag = 'tests'

mlp_param_grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive']
}
random_forest_params = {"n_estimators": [500, 1000, 1500],
                        "min_samples_split": [3, 4, 5, 10],
                        "min_samples_leaf": [3, 4, 5, 10]
                        }


RFregressor = MLPRegressor()
mlp_regressor = RandomForestRegressor()

# first tries
first_params = ["c_sp_fao", "SST", "d13C_cor", 'c_ocean']

# saved variables in read_data
use_params = ["d13C_cor", "c_sp_fao", "c_ocean", "sampling_date", "length_cm", "d13C_pm", "sample_year", "d13C_cor",
              "SST", "MLD", "Chl.a", "NPP", "d20", "d18", "d12", "O2_375m", "d13Cdic",
              "d13Cpom", "d15Npom", "doxycline", "dnethetero"]

coloring_vars = ["region", "region.col", "lon_dec", "lat_dec", "sample_owner"]

