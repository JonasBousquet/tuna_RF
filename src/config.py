from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Path
path_to_file = '../data'
run_tag = 'RF_RFvars'


# Regressor type
RFregressor = RandomForestRegressor()
#random_forest_params = {"n_estimators": [500, 1000, 1500],
#                        "min_samples_split": [3, 4, 5, 10],
#                        "min_samples_leaf": [3, 4, 5, 10]
#                        }

random_forest_params = {"n_estimators": [500],
                        "min_samples_split": [3],
                        "min_samples_leaf": [3]}

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
coloring_vars = ["region", "region.col", "lon_dec", "lat_dec", "sample_owner", "d13C_pm"]


RFvars = ["sample_year", "d13Cdic", "SST", "d13C_cor"]

allvars = ["d13C_cor", "c_sp_fao", "c_ocean", "length_cm", "sample_year",
              "SST", "MLD", "Chl.a", "NPP", "d20", "d18", "d12", "O2_375m", "d13Cdic",
              "d13Cpom", "d15Npom", "doxycline", "dnethetero"]

GAMvars = ["d13C_cor", "length_cm", 'sample_year', 'd13Cdic', 'd13Cpom', 'd15Npom',
             'SST', "MLD", 'd12', "O2_375m", "doxycline"]

REDvars = ['SST', "Chl.a", "length_cm", 'MLD']

RED2vars = ['SST', "Chl.a", "length_cm"]
TESTvars = ['SST', 'MLD', 'Chl.a']

# variables = {'RFvars': RFvars, 'allvars': allvars, 'GAMvars': GAMvars, 'REDvars': REDvars, 'RED2vars': RED2vars}

variables = {'test_vars': TESTvars}