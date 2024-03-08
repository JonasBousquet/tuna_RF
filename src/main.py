from sklearn.model_selection import (GridSearchCV)
import config
import utils
import preprocessor as pre
import plots

if config.Intel_patch:
    from sklearnex import patch_sklearn
    patch_sklearn()

def main(data_path: str,
         target: str,
         model,
         model_param_grid: dict,
         test_size: float,
         random_state: int,
         cv: int,
         variables: list,
         run_tag: str):

    error_dir, importance_dir, main_dir, val_curves_dir, model_dir, plot_dir = utils.generate_run_directories(
        tag=run_tag)

    # Load the data
    X_train, X_test, y_train, y_test = pre.choose_data(data_path=data_path,
                                                       target=target,
                                                       variables=variables)

    # Encode the data
    X_train, X_test, encoder_dict = utils.encode_data(X_train, X_test)
    # Change date into year and numeric
    if config.date_to_year:
        X_train = pre.date_to_year(X_train, 'sample_year')
        X_test = pre.date_to_year(X_test, 'sample_year')

    # Create a grid search
    grid_search = GridSearchCV(model,
                               model_param_grid,
                               cv=cv,
                               verbose=3,
                               return_train_score=True,
                               n_jobs=-1).fit(X_train, y_train)

    # Fit the grid search to the data
    best_estimator = grid_search.best_estimator_
    utils.save_model(model=best_estimator, path=f"{model_dir}{model.__class__.__name__}.pkl")
    utils.save_params(logdir=f"{main_dir}/models/",
                      filename=model.__class__.__name__,
                      params=grid_search.best_params_)

    # Print the best parameters
    utils.console.log(grid_search.best_params_)

    # Print the best score
    utils.console.log(f"Best score: {grid_search.best_score_}")

    # Predict the target
    y_pred = grid_search.predict(X_test)

    # Plots
    plots.pred_vs_real(y_pred, y_test, plot_dir, run_tag)
    plots.live_feature_importance(best_estimator, plot_dir, run_tag, encoder_dict)

    utils.print_regression_metrics(y_test, y_pred)

    utils.console.save_text(f"{main_dir}/run_log.txt")


if __name__ == '__main__':
    for i in config.variables:
        run_tag = i
        variables = config.variables[i]
        main(data_path=config.path_to_file,
             target=config.target,
             model=config.RFregressor,
             model_param_grid=config.random_forest_params,
             test_size=0.2,
             random_state=42,
             cv=5,
             variables=variables,
             run_tag=run_tag)
