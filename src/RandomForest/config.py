from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------------------------------------------------------
# --------------------------------------- Paths 'n stuff --------------------------------------

# Path
path_to_file = '../data'

# only used if fixed_train_test_data = True
name_of_file = 'db_d13c_sorted_utf.csv'
# name_of_file = 'mercury_data.csv'
target = 'd13C_cor'
# target = 'logHg'

# ------------------------------------------------------------------------------------------------------------
# --------------------------------------- Regressor params --------------------------------------
RFregressor = RandomForestRegressor()
random_forest_params = {"n_estimators": [500, 1000, 1500],
                        "min_samples_split": [3, 4, 5, 10],
                        "min_samples_leaf": [3, 4, 5, 10]
                        }

# Testing params
test_forest_params = {"n_estimators": [500],
                      "min_samples_split": [3],
                      "min_samples_leaf": [3]}

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

allvars = ["c_sp_fao", "c_ocean", "length_cm", "sample_year",
           "SST", "MLD", "Chl.a", "NPP", "d20", "d18", "d12", "O2_375m", "d13Cdic",
           "d13Cpom", "d15Npom", "doxycline", "dnethetero"]

GAMvars = ["length_cm", 'sample_year', 'd13Cdic', 'd13Cpom', 'd15Npom',
           'SST', "MLD", 'd12', "O2_375m", "doxycline"]

REDvars = ['SST', "Chl.a", "length_cm", 'MLD']

RED2vars = ['SST', "Chl.a", "length_cm"]

TESTvars = ['SST',  'Chl.a', 'MLD']
e
HgVars = ['length_cm', 'Hg0_6x7.5', 'dnethetero', 'O2_375m']
HgVars2 = ["length_cm", 'sample_year', 'd13Cdic', 'd13Cpom', 'd15Npom',
           'SST', "MLD", 'd12', "O2_375m", "doxycline", 'Hg0_6x7.5']
# ----------------------------------- Variables to use in the model with runtag as [key]--------------------------------
# variables = {'RFvars': RFvars, 'allvars': allvars, 'GAMvars': GAMvars, 'REDvars': REDvars, 'RED2vars': RED2vars}
variables = {'test_vars': allvars}

