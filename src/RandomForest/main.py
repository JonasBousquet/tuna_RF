import plots
import config
import preprocessor as pre
from sklearn.model_selection import (GridSearchCV)
from utils import generate_run_directories, console, save_model, save_params, print_regression_metrics


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

    error_dir, importance_dir, main_dir, val_curves_dir, model_dir, plot_dir = generate_run_directories(
        tag=run_tag)

    # Load the data
    X_train, X_test, y_train, y_test = pre.choose_data(data_path=data_path,
                                                       target=target,
                                                       variables=variables)

    # Encode the data
    X_train, X_test, encoder_dict = pre.encode_data(X_train, X_test)
    # Change date into year and numeric
    if config.date_to_year:
        X_train = pre.date_to_year(X_train, 'sample_year')
        X_test = pre.date_to_year(X_test, 'sample_year')

    # Create a grid search
    grid_search = GridSearchCV(model,
                               model_param_grid,
                               cv=cv,
                               verbose=1,
                               return_train_score=True,
                               n_jobs=-1).fit(X_train, y_train)

    # Fit the grid search to the data
    best_estimator = grid_search.best_estimator_
    save_model(model=best_estimator, path=f"{model_dir}{model.__class__.__name__}.pkl")
    save_params(logdir=f"{main_dir}/models/",
                      filename=model.__class__.__name__,
                      params=grid_search.best_params_)

    # Print the best parameters
    console.log(grid_search.best_params_)

    # Print the best score
    console.log(f"Best score: {grid_search.best_score_}")

    # Predict the target
    y_pred = grid_search.predict(X_test)

    # Plots
    plots.pred_vs_real(y_pred=y_pred,
                            y_test=y_test,
                            plotdir=plot_dir,
                            runtag=run_tag,
                            target=target)
    plots.live_feature_importance(model=best_estimator,
                                  plot_dir=plot_dir,
                                  runtag=run_tag,
                                  encoder=encoder_dict)

    print_regression_metrics(y_test, y_pred)

    console.save_text(f"{main_dir}/run_log.txt")


if __name__ == '__main__':
    for i in config.variables:
        run_tag = i
        variables = config.variables[i]
        main(data_path=config.path_to_file,
             target=config.target,
             model=config.RFregressor,
             model_param_grid=config.test_forest_params if config.test else config.RFregressor,
             test_size=0.2,
             random_state=42,
             cv=5,
             variables=variables,
             run_tag=run_tag)
