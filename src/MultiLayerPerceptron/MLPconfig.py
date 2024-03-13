from sklearn.neural_network import MLPRegressor
# ------------------------------------------------------------------------------------------------------------
# --------------------------------------- Paths 'n stuff --------------------------------------

# Path
path_to_file = '../data'

# only used if fixed_train_test_data = False
name_of_file = 'db_d13c_sorted_utf.csv'
target = 'd13C_cor'

# ------------------------------------------------------------------------------------------------------------
# --------------------------------------- Regressor params --------------------------------------
mlp_regressor = MLPRegressor()
mlp_param_grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive']
}

test_params = {
    'hidden_layer_sizes': [(50, 50, 50)],
    'activation': ['tanh'],
    'solver': ['sgd'],
    'alpha': [0.0001],
    'learning_rate': ['constant']}
# ------------------------------------------------------------------------------------------------------------
# --------------------------------------- I dont know how to call that yet --------------------------------------

date_to_year = False
fixed_train_test_data = True
Intel_patch = True  # Used to speed up calculation time, WORKS ONLY ON INTEL BASED SYSTEMS
test = True

# ------------------------------------------------------------------------------------------------------------
# --------------------------------------- Variables as lists --------------------------------------

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

# ----------------------------------- Variables to use in the model with runtag as [key]--------------------------------
# variables = {'RFvars': RFvars, 'allvars': allvars, 'GAMvars': GAMvars, 'REDvars': REDvars, 'RED2vars': RED2vars}
variables = {'test_vars': TESTvars}
