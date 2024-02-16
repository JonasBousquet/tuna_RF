from sklearn.neural_network import MLPRegressor

mlp_param_grid = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

mlp_regressor = MLPRegressor()

# first tries
first_params = ["c_sp_fao", "SST", "d13C_cor"]

# saved variables in read_data
use_params = ["d13C_cor", "c_sp_fao", "c_ocean", "sampling_date", "length_cm", "d13C_pm", "sample_year", "d13C_cor",
              "SST", "MLD", "Chl.a", "NPP", "d20", "d18", "d12", "O2_375m", "d13Cdic",
              "d13Cpom", "d15Npom", "doxycline", "dnethetero"]

coloring_vars = ["region", "region.col", "lon_dec", "lat_dec", "sample_owner"]

