import plots
import utils
import config

def main(runname):

    # load model
    complete_dir, model_dir, plot_dir = utils.paths(runname)
    model, vars_in = utils.load_model(model_dir)
    X_test, y_test = utils.load_train_data('../data', vars_in)
    # calculations

    # plots



    debuggg = 'just a var for debug purpose'




if __name__ == '__main__':
    main(runname=config.run_name)