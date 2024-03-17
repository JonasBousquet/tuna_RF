import os
import plots
import utils
import config

def main(runname):

    # load model
    complete_dir, model_dir, plot_dir, runtag = utils.paths(runname)
    model, vars_in = utils.load_model(model_dir)
    X_test, y_test, X_train, y_train = utils.load_data('../data', vars_in)
    y_pred, metrics = utils.model_fit(model=model,
                                      X_test=X_test,
                                      y_test=y_test,
                                      X_train=X_train,
                                      y_train=y_train)


    # calculations
    utils.shap_values(model=model,
                      X_test=X_test,
                      runtag=runtag,
                      plot_dir=plot_dir)
    # plots



    debuggg = 'just a var for debug purpose'




if __name__ == '__main__':
    if config.loop:
        for i in os.listdir(config.saved_path):
            print(f'Running model {i}')
            main(runname=i)
    else:
        main(runname=config.run_name)