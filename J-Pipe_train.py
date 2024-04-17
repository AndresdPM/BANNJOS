"""

"""

import sys, os
import pandas as pd
import polars as pl
import numpy as np
import codecs, json
from multiprocessing import Pool, cpu_count
import glob
import importlib
import argparse
from datetime import datetime
import time
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

def J_pipe(argv):

    examples = '''Examples:

    BANNJOS --experiment_name "Classify"

    BANNJOS --experiment_name "Classify" --model_nominal "dropout"

    BANNJOS --experiment_name "Classify" --model_nominal "deterministic" --layers_nominal 300 100 50

    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, usage='%(prog)s [options]', description='GaiaHub computes proper motions (PM) combining HST and Gaia data.', epilog=examples)

    # Search options
    parser.add_argument('--experiment_name', type=str, default = None, help='Experiment name.')
    parser.add_argument('--training_catalog', type=str, default = './used_catalogs/J-PLUS_SPECLASS.csv', help='Training catalog.')
    parser.add_argument('--output_path', type=str, default = './results/', help='Output path.')
    parser.add_argument('--saved_pdfs_catalog', type=str, default = None, help='Training catalog.')
    parser.add_argument('--training_flags', type=str, default = 'SDSS', help='Flags for quality selection. Otions are "SDSS", "GAIA", "LAMOST_stellar"')
    parser.add_argument('--y_vars', type=str, nargs='+', default= ['CLASS'], help='Target variable(s).')
    parser.add_argument('--y_vars_err', type=str, nargs='+', default= [], help='Target variable(s) uncertainties.')
    parser.add_argument('--each_nrows', type=int, default = 1, help='Read the training file each n rows.')
    parser.add_argument('--nrows', type=int, default = None, help='Read the training file the first n rows.')
    parser.add_argument('--skip_nrows', type=int, default = None, help='Skip the first n rows from the training file. Useful when in combination with each_nrows to get complementary sets.')
    parser.add_argument('--model_nominal', type=str, default = 'dropout', help='Model name.')
    parser.add_argument('--layers_nominal', type=int, nargs='+', default= [500, 1000, 700, 300], help='Layer configuration.')
    parser.add_argument('--dropout_nominal', type=float, nargs='+', default= [0.2, 0.2, 0.2, 0.2], help='Layer dropout configuration.')
    parser.add_argument('--input_dropout_nominal', type=float, default= 0.1, help='Dropout ratio at the input layer')
    parser.add_argument('--loss_nominal', type=str, default = 'mean_squared_error', help='Loss function.')
    parser.add_argument('--batch_size_nominal', type=int, default = 128, help='Bach size.')
    parser.add_argument('--epochs_nominal', type=int, default = 5000, help='Max number of epochs.')
    parser.add_argument('--early_stop_patience_nominal', type=int, default = 300, help='Patience before stoping training.')
    parser.add_argument('--nominal_model_mode', type=str, nargs='+', default = ['read', 'train', 'predict', 'plot'], help='Train the model or just load the weights. Options are "read", "train", "predict", "plot".')
    parser.add_argument('--nominal_alleatoric_n_iter', type=int, default = 3000, help='Number of realizations of the posterior')
    parser.add_argument('--full_cov', action='store_true', help='Compute the complete covariance.')
    parser.add_argument('--verbose', type=int, default = 1, help='Verbosity level.')
    parser.add_argument('--n_chunks', type=int, default = 15, help='Divide the work in n chunks.')
    parser.add_argument('--multiprocesing', action='store_true', help='Use multiprocesing.')
    parser.add_argument('--save_training_catalog', action='store_true', help='Save the preprocessed catalog')
    parser.add_argument('--n_processors', type=int, default = 15, help='Number of processors used if multiprocessing in use.')
    parser.add_argument('--parameters_file', type=str, default = None, help='Alternative json file containing parameters.')
    parser.add_argument('--validation_folder', type=str, default = None, help='Alternative validation files path.')
    parser.add_argument('--initial_learning_rate', type=float, default= 1e-4, help='Ininitial learning rate')
    parser.add_argument('--step_decay_learning_rate', type=int, default= 30, help='Step decay learning rate')

    args = parser.parse_args(argv)

    import return_experiment_params as rep
    importlib.reload(rep)

    print(args)

    params = rep.return_experiment_params(args)

    if params['experiment']['multiprocesing']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        pool = Pool(cpu_count()-1)
    else:
        pool = None

    # We import all the libraries with the proper CUDA configuration
    import libraries as lib
    importlib.reload(lib)

    output_directory = params['variables_n_preproc']['output_path']+params['experiment_name']

    lib.create_dir(output_directory+'/')
    lib.create_dir(output_directory+'/pdfs/')
    lib.create_dir(output_directory+'/model/')
    lib.create_dir(output_directory+'/losses/')

    if 'read' in params['model_nominal']['mode']:

        X, eX, y, ey, y_cat, output, quality_data, features_names, pre_targets, targets_error_names, test_X, test_X_error, test_output, norm_minimum, norm_dynamic_range, fluxes_zpt = lib.get_data(params['variables_n_preproc'])

        # We select variables with some variance
        variance_features = VarianceThreshold().fit(X).get_feature_names_out()
        variance_features_indexes = X.columns.get_indexer(variance_features)
        X = X.loc[:, variance_features]
        eX = eX.iloc[:, variance_features_indexes]
        features_names = list(variance_features)

        X_train, X_train_error, y_train, y_train_error, y_cat_train, targets_names, X_valid, X_valid_error, y_valid, y_valid_error, y_cat_valid, valid_output = lib.get_train_val(X, eX, y, ey, y_cat, output, params['variables_n_preproc'])

        if params['experiment']['perform_RFE']:
            lib.create_dir(output_directory+'/rfecv/')
            best_features, ranking_all_features, rfe_cv_result = lib.RFE_selector(X_train, y_train, params['model_nominal'], scoring = 'roc_auc', min_features_to_select = int(len(features_names)/20), n_splits=5, step = int(len(features_names)/50), output_name =output_directory+'/rfecv/rfecv')
        elif params['experiment']['perform_ANOVA']:
            from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
            f_regression_filter = SelectKBest(f_regression, k=int(2*X_train.shape[1]/3))
            mutual_info_regression_filter = SelectKBest(mutual_info_regression, k=int(2*X_train.shape[1]/3))

            best_features = []
            for target_name in targets_names[:-1]:
                best_features.append(list(f_regression_filter.fit(X_train, y_train.loc[:, target_name]).get_feature_names_out()))
                best_features.append(list(mutual_info_regression_filter.fit(X_train, y_train.loc[:, target_name]).get_feature_names_out()))

            best_features = list(set([item for row in best_features for item in row]))
        else:
            best_features = X_train.columns.to_list()
        best_features_indexes = X_train.columns.get_indexer(best_features)
        X_train_best =  X_train.loc[:, best_features]
        X_train_best_error = X_train_error.iloc[:, best_features_indexes]
        X_valid_best = X_valid.loc[:, best_features]
        X_valid_best_error = X_valid_error.iloc[:, best_features_indexes]

        del [X_train, X_train_error, X_valid, X_valid_error]

        # We balance the sample
        if params['experiment']['Balance_training'] is not None:
            print('Before balancing, the distribution of classes is as follow:\n', y_train.sum(axis = 0))
            if params['experiment']['Balance_training'] == 'SMOTE':
                print('Applying SMOTE now...')
                sm = SMOTE(random_state=42, n_jobs=-2)
                X_train_best, y_res = sm.fit_resample(X_train_best, y_cat_train.CLASS_num)
                X_train_best_error, y_res = sm.fit_resample(X_train_best_error, y_cat_train.CLASS_num)
                y_train, y_res = sm.fit_resample(y_train, y_cat_train.CLASS_num)
                try:
                    y_train_error, y_res = sm.fit_resample(y_train_error, y_cat_train.CLASS_num)
                except:
                    pass
            elif params['experiment']['Balance_training'] == 'undersample':
                ## UNDERSAMPLE
                print('Applying Undersampling...')
                grouped_by_class = X_train_best.join(y_cat_train).groupby(['CLASS'])
                n_samples = grouped_by_class['CLASS'].count().min()
                X_train_best = grouped_by_class.sample(n_samples, replace=True, random_state = params['variables_n_preproc']['random_generator_seed']).drop(columns = y_cat_train.columns.to_list())
                X_train_best_error = X_train_best_error.sample(n_samples, replace=True, random_state = params['variables_n_preproc']['random_generator_seed'])

            print('After balancing, the distribution of classes is as follow:\n', y_train.sum(axis = 0))

        if params['variables_n_preproc']['save_training_catalog']:
            X_train_best.to_csv('./used_catalogs/X_train_best.csv', index = False)
            X_train_best_error.to_csv('./used_catalogs/X_train_best_error.csv', index = False)
            y_train.to_csv('./used_catalogs/y_train.csv', index = False)
            try:
                y_train_error.to_csv('./used_catalogs/y_train_error.csv', index = False)
            except:
                pass

        if params['variables_n_preproc']['save_validation_catalog']:
            lib.create_dir(output_directory+'/validation/')
            X_valid_best.to_csv('%s/validation/validation_X.csv'%output_directory, index=False)
            X_valid_best_error.to_csv('%s/validation/validation_X_error.csv'%output_directory, index=False)
            y_valid.to_csv('%s/validation/validation_y.csv'%output_directory, index=False)
            y_valid_error.to_csv('%s/validation/validation_y_error.csv'%output_directory, index=False)
            valid_output.to_csv('%s/validation/validation_extra.csv'%output_directory, index=False)

        n_inputs_nominal, n_instances_nominal, n_outputs_nominal = X_train_best.shape[1], X_train_best.shape[0], y_train.shape[1]

        lib.dict_merge(params, {'variables_n_preproc':{'features_names':features_names, 'targets_names':targets_names, 'targets_error_names':targets_error_names, 'norm_minimum':norm_minimum.tolist(), 'norm_dynamic_range':norm_dynamic_range.tolist(), 'fluxes_zpt': fluxes_zpt, 'best_features':best_features},
        'model_nominal':{'targets_names':['pred_nom_%s'%name for name in targets_names], 'pre_trained_n_inputs':n_inputs_nominal, 'pre_trained_n_instances':n_instances_nominal, 'pre_trained_n_outputs':n_outputs_nominal}})

        # create json object from dictionary
        json_params = json.dumps(params)

        # open file for writing, "w"
        f = open('%s/parameters.json'%output_directory,'w')
        f.write(json_params)
        f.close()

    if 'train' in params['model_nominal']['mode']:

        if 'read' not in params['model_nominal']['mode']:

            if args.parameters_file is not None:
                with open(args.parameters_file) as json_file:
                    params_pre = json.load(json_file)
                params_pre['experiment_name'] = params['experiment_name']
                params_pre['model_nominal'].update(params['model_nominal'])
                params_pre['model_variance'].update(params['model_variance'])
                params = params_pre
                del [params_pre]
            else:
                with open('%s/parameters.json'%output_directory) as json_file:
                    params = json.load(json_file)

            print('Reading pre-processed files...')
            X_train_best = pl.read_csv('./used_catalogs/X_train_best.csv').to_pandas()
            X_train_best_error = pl.read_csv('./used_catalogs/X_train_best_error.csv').to_pandas()
            y_train = pl.read_csv('./used_catalogs/y_train.csv').to_pandas()
            try:
                y_train_error = pl.read_csv('./used_catalogs/y_train_error.csv').to_pandas()
            except:
                pass

        if params['experiment']['fit_uncertainty_cv']:
            if params['experiment']['perform_cv']:
                # We are going to estimate the variance of our model by fitting it to our data with CV
                print('')
                print('Fit cross validation\n')

                if params['stratify_var'] is not None:
                    stratify = y_cat_train.loc[:, params['stratify_var']]
                else:
                    stratify = None

                cv_indices, results_cv, losses_cv = lib.run_experiment_cv(params['model_nominal'], X_train_best, y_train, X_error= X_train_best_error, y_error = y_train_error, n_splits=params['experiment']['nsplits_cv'], n_repeats=params['experiment']['nrepeats_cv'], stratify = stratify, n_chunks = params['experiment']['n_chunks'], pool = pool)

                losses_cv.to_csv('%s/losses/loss_cv_%s.csv'%(output_directory, params['model_nominal']['model']))

                obs_variance_cv = (results_cv.loc[:, ['%s_median'%col for col in params['model_nominal']['targets_names']]].values - results_cv.loc[:, params['targets_names']]).add_prefix('var_').abs()

                results_cv = results_cv.join(obs_variance_cv)

                results_cv.to_csv('%s/results_cv.csv'%output_directory)
                pd.Series(data=cv_indices, name = 'cv_indices').to_csv('%s/indices_cv.csv'%output_directory)

                del [results_cv]
            else:
                results_cv = pl.read_csv('%s/results_cv.csv'%output_directory, index_col = [0]).to_pandas()
                obs_variance_cv = results_cv.loc[:, [col for col in results_cv if 'var_' in col]]
                del [results_cv]
                cv_indices = list(pd.read_csv('%s/indices_cv.csv'%output_directory, index_col = [0]).values.flatten())

            X_train_best_variance = X_train_best.iloc[cv_indices, :]

            n_inputs_variance, n_instances_variance, n_outputs_variance = X_train_best_variance.shape[1], X_train_best_variance.shape[0], obs_variance_cv.shape[1]

            lib.dict_merge(params, {'model_variance':{'targets_names':['pred_var_%s'%name for name in  targets_names], 'pre_trained_n_inputs':n_inputs_variance, 'pre_trained_n_instances':n_instances_variance, 'pre_trained_n_outputs':n_outputs_variance}})

            # Create model for the variance
            print('')
            print('Fit variance\n')

            model_variance_built, losses_variance = lib.build_model(params['model_variance'], X_train = X_train_best_variance, y_train = obs_variance_cv)

            losses_variance.to_csv('%s/losses/losses_variance_%s.csv'%(output_directory, params['model_variance']['model']))
            model_variance_built.save_weights(params['model_variance']['pre_trained_weights_file'])
        else:
            losses_variance = None

        # Lets control the training time
        start_time = datetime.now()
        start_time_s = time.time()

        # Now we can train the nominal model with the entire set:
        print('')
        print('Fit all data\n')

        model_nominal_built, losses_nominal = lib.build_model(params['model_nominal'], X_train = X_train_best, y_train = y_train)

        losses_nominal.to_csv('%s/losses/losses_nominal_%s.csv'%(output_directory, params['model_nominal']['model']))
        model_nominal_built.save_weights(params['model_nominal']['pre_trained_weights_file'])

        # create json object from dictionary
        json_params = json.dumps(params)

        # open file for writing, "w"
        f = open('%s/parameters.json'%output_directory,'w')
        f.write(json_params)
        f.close()

        del [X_train_best, X_train_best_error, y_train]
        try:
            del [y_train_error]
        except:
            pass

    if 'predict' in params['model_nominal']['mode']:

        if 'read' not in params['model_nominal']['mode']:

            if args.validation_folder is not None:
                X_valid_best = pl.read_csv('%s/validation_X.csv'%args.validation_folder).to_pandas()
                y_valid = pl.read_csv('%s/validation_y.csv'%args.validation_folder).to_pandas()
                valid_output = pl.read_csv('%s/validation_extra.csv'%args.validation_folder).to_pandas()
            else:
                X_valid_best = pl.read_csv('%s/validation/validation_X.csv'%output_directory).to_pandas()
                y_valid = pl.read_csv('%s/validation/validation_y.csv'%output_directory).to_pandas()
                valid_output = pl.read_csv('%s/validation/validation_extra.csv'%output_directory).to_pandas()

            try:
                y_valid_error = pl.read_csv('%s/validation/validation_y_error.csv'%output_directory).to_pandas()
            except:
                y_valid_error = None
            try:
                X_valid_best_error = pl.read_csv('%s/validation/validation_X_error.csv'%output_directory).to_pandas()
            except:
                X_valid_best_error = None

            with open('%s/parameters.json'%output_directory) as json_file:
                params = json.load(json_file)

            params['model_nominal']['alleatoric_n_iter'] = args.nominal_alleatoric_n_iter
            params['model_nominal']['full_cov'] = args.full_cov
            params['experiment']['n_chunks'] = args.n_chunks
            params['experiment']['n_processors'] = args.n_processors

            model_nominal_built = None
            test_X = None

        # Indices to save PDFs
        if args.saved_pdfs_catalog is not None:
            params['variables_n_preproc']['saved_pdfs_indexes'] = pl.read_csv(args.saved_pdfs_catalog)['Index'].to_list()
        else:
            params['variables_n_preproc']['saved_pdfs_indexes'] = np.linspace(0, len(y_valid)-1, 15, dtype = int)

        print('')
        print('Predict all data\n')

        predictions_nominal, y_pred_nominal = lib.predict_validate(params['model_nominal'], X_valid_best, y_valid, X_valid_error = X_valid_best_error, y_valid_error = y_valid_error, built_model = model_nominal_built, idxs_pred = params['variables_n_preproc']['saved_pdfs_indexes'], n_chunks = params['experiment']['n_chunks'], pool = pool)

        # Now we print the training time.
        try:
            end_time = datetime.now()
            print('')
            print('Training duration: {}'.format(end_time - start_time))
            lib.dict_merge(params, {'experiment':{'execution_time':float(time.time() - start_time_s)}})

            # create json object from dictionary
            json_params = json.dumps(params)

            # open file for writing, "w"
            f = open('%s/parameters.json'%output_directory,'w')
            f.write(json_params)
            f.close()
        except:
            pass

        non_common_cols = [col for col in predictions_nominal.columns if col not in valid_output.columns]

        if params['experiment']['fit_uncertainty_cv']:
            # The real observed variance in the validation set is:
            obs_variance_val = (predictions_nominal.loc[:, ['%s_median'%col for col in params['model_nominal']['targets_names']]].values - predictions_nominal.loc[:, params['targets_names']]).add_prefix('var_').abs()

            # We also make predictions on the errorbars for the validation set:
            print('')
            print('Predict all data uncertainties\n')
            predictions_sigma, y_pred_sigma = lib.predict_validate(params['model_variance'], X_valid_best, obs_variance_val, X_valid_error = X_valid_best_error, y_valid_error = None, built_model = model_variance_built, n_chunks = params['experiment']['n_chunks'], pool = pool)

            # Save the results
            predictions_nominal = valid_output.join(predictions_nominal.loc[:, non_common_cols]).join(predictions_sigma.clip(lower=1e-4))
        else:
            predictions_nominal = valid_output.join(predictions_nominal.loc[:, non_common_cols])

        pl.from_pandas(predictions_nominal).write_csv('%s/results.csv'%output_directory, float_precision=6)
        #predictions_nominal.to_csv('%s/results.csv'%output_directory, index=False, float_format='%.8f')

        # We save the pdfs
        pdfs = []
        for ii, idx in enumerate(params['variables_n_preproc']['saved_pdfs_indexes']):
            try:
                pdf = y_pred_nominal[:, ii, :]
                pdf = pd.DataFrame(columns = ['pred_nom_%s'%col for col in y_valid.columns.to_list()], data=pdf)
                pdfs.append(pdf)
                pdf.to_csv('%s/pdfs/y_pred_%i.csv'%(output_directory, idx), index = False, float_format='%.8f')
            except:
                pass

        # Here we predict for the external table if exist
        if test_X is not None:
            test_X_best = test_X.loc[:, best_features]
            test_X_best_error = test_X_error.iloc[:, best_features_indexes]

            print('')
            print('Predict test data\n')
            predictions_test, y_pred_test = lib.predict(params['model_nominal'], test_X_best, X_error = test_X_best_error, idxs_pred = None, n_chunks = params['experiment']['n_chunks'], pool = pool)

            non_common_cols = [col for col in predictions_test.columns if col not in test_output.columns]

            if params['experiment']['fit_uncertainty_cv']:
                print('')
                print('Predict test data uncertainties\n')
                predictions_test_sigma, y_pred_test_sigma = lib.predict(params['model_variance'], test_X_best, X_error = None, n_chunks = params['experiment']['n_chunks'], pool = pool)

                predictions_test = test_output.join(predictions_test.loc[:, non_common_cols]).join(predictions_test_sigma.clip(lower=1e-4))
            else:
                predictions_test = test_output.join(predictions_test.loc[:, non_common_cols])

            # Save the results
            predictions_test.to_csv('%s/results_test.csv'%output_directory, index=False, float_format='%.8f')

    if 'plot' in params['model_nominal']['mode']:

        if 'read' not in params['model_nominal']['mode']:

            with open('%s/parameters.json'%output_directory) as json_file:
                params = json.load(json_file)

        if 'predict' not in params['model_nominal']['mode']:
            predictions_nominal = pl.read_csv('%s/results.csv'%output_directory).to_pandas()
            pdfs = []
            for ii, idx in enumerate(params['variables_n_preproc']['saved_pdfs_indexes']):
                try:
                    pdfs.append(pl.read_csv('%s/pdfs/y_pred_%i.csv'%(output_directory, idx)).to_pandas())
                except:
                    pass

        if 'train' not in params['model_nominal']['mode']:
            losses_nominal = pl.read_csv('%s/losses/losses_nominal_%s.csv'%(output_directory,  params['model_nominal']['model'])).to_pandas()
            if params['experiment']['fit_uncertainty_cv']:
                losses_variance = pl.read_csv('%s/losses/losses_variance_%s.csv'%(output_directory,  params['model_variance']['model'])).to_pandas()
            else:
                losses_variance = None

        lib.plot_predictions_nominal(predictions_nominal, losses_nominal, pdfs, params, losses_variance = losses_variance)

    else:
        print('Only three modes are supported: train, load, and plot')

    try:
        pool.close()
    except:
        pass

if __name__ == '__main__':

    J_pipe(sys.argv[1:])
    sys.exit(0)

"""
Andres del Pino Molina
"""

