from sklearn.ensemble import RandomForestRegressor
# ------------------------------------------------------------------------------------------------------------
# --------------------------------------- Paths 'n stuff --------------------------------------

# Path
path_to_file = './data'

# only used if fixed_train_test_data = True
name_of_file = 'db_d13c_sorted_utf.csv'
target = 'd13C_cor'

# ------------------------------------------------------------------------------------------------------------
# --------------------------------------- Regressor params --------------------------------------
RFregressor = RandomForestRegressor()
#random_forest_params = {"n_estimators": [500, 1000, 1500],
#                        "min_samples_split": [3, 4, 5, 10],
#                        "min_samples_leaf": [3, 4, 5, 10]
#                        }

# Testing params
random_forest_params = {"n_estimators": [500],
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
