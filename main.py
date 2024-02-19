import pandas as pd
from sklearn.model_selection import (GridSearchCV, train_test_split, cross_validate)
import config
import utils
import preprocessor as pre


def main(data_path: str,
         target: str,
         model,
         model_param_grid: dict,
         test_size: float,
         random_state: int,
         cv: int):

    error_dir, importance_dir, main_dir, val_curves_dir, model_dir = utils.generate_run_directories(tag=config.run_tag)

    # Load the data
    data = pre.load_data(config.path_to_file, config.first_params)

    # Encode species name
    data = pre.one_hot(data, 'c_sp_fao')

    # Split the data into features and target
    X = data.drop(target, axis=1)
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    # Create a grid search
    grid_search = GridSearchCV(model,
                               model_param_grid,
                               cv=cv,
                               verbose=3,
                               return_train_score=True,
                               n_jobs=-1).fit(X_train, y_train)

    # Fit the grid search to the data
    best_estimator = grid_search.best_estimator_
    utils.save_model(model=best_estimator, path=model_dir + f"{model.__class__.__name__}.pkl")
    utils.save_params(logdir=main_dir + "/models/", filename=model.__class__.__name__,
                      params=grid_search.best_params_)

    # Print the best parameters
    utils.console.log(grid_search.best_params_)

    # Print the best score
    utils.console.log(grid_search.best_score_)

    # Predict the target
    y_pred = grid_search.predict(X_test)

    utils.print_regression_metrics(y_test, y_pred)

    # Print the cross validation scores
    utils.console.log(cross_validate(best_estimator, X, y, cv=cv, scoring=('r2',
                                                                           'max_error')
                         ))

    utils.console.save_text(main_dir + "/run_log.txt")


if __name__ == '__main__':
    main(data_path=config.path_to_file,
         target='d13C_cor',
         model=config.RFregressor,
         model_param_grid=config.random_forest_params,
         test_size=0.2,
         random_state=42,
         cv=5)

