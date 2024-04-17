import sys, os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import polars as pl
import numpy as np
import codecs, json
from multiprocessing import Pool, cpu_count
import importlib
import time
from datetime import timedelta
import argparse


def remaining_time_calculator(start_time, current_time, current_file, total_files):

    average_time_per_file = (current_time-start_time)/current_file

    remaining_time = (total_files - current_file) * average_time_per_file

    td_str = str(timedelta(seconds=remaining_time))

    # split string into individual component
    x = td_str.split(':')

    return 'Approximate remaining time:'+ x[0]+ ' hours '+ x[1] + ' minutes '+ x[2] + ' seconds'


def J_predict(argvs):


    examples = '''Examples:

    BANNJOS_predict experiment_name "Classify"

    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, usage='%(prog)s [options]', description='GaiaHub computes proper motions (PM) combining HST and Gaia data.', epilog=examples)

    # Search options
    parser.add_argument('experiment_name', type=str, default = None, help='Experiment name.')
    parser.add_argument('data_files', type=str, default = None, help='List with the files where to make predictions.')
    parser.add_argument('--output_directory', type=str, default = './results/', help='Output path.')
    parser.add_argument('--alleatoric_n_iter', type=int, default = 300, help='Number of realizations of the posterior')
    parser.add_argument('--full_cov', action='store_true', help='Compute the complete covariance.')
    parser.add_argument('--verbose', type=int, default = 1, help='Verbosity level.')
    parser.add_argument('--n_chunks', type=int, default = 14, help='Divide the work in n chunks.')
    parser.add_argument('--multiprocesing', action='store_true', help='Use multiprocesing.')
    parser.add_argument('--write_test_output', action='store_true', help='Whether to include extra columns in the output for later tests.')
    parser.add_argument('--incremental', action='store_true', help='Incremental results. Useul if something goes wrong and you want to pick up the process where it failed.')
    parser.add_argument('--external_cat', type=str, default = None, help='External catalog to be crossmatched before predicting.')
    parser.add_argument('--dtypes_table', type=str, default = '../../GetJ-Plus/output_final/J-PLUS_DR3_86109.csv', help='External catalog that contains the dtypes to be used for all files')

    args = parser.parse_args(argvs)

    with open(args.data_files) as f:
        data_files = [line.rstrip() for line in f]

    with open('./results/%s/parameters.json'%args.experiment_name) as json_file:
        params = json.load(json_file)

    # If you want to change the presictions parameters, it's now or never!
    params['model_nominal']['targets_names'] = [('%s'%name).replace('pred_nom_', '') for name in params['model_nominal']['targets_names']]
    params['experiment']['multiprocesing'] = args.multiprocesing
    params['experiment']['n_chunks'] = args.n_chunks
    params['model_nominal']['alleatoric_n_iter'] = args.alleatoric_n_iter
    params['model_nominal']['full_cov'] = args.full_cov

    if params['experiment']['multiprocesing']:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To remove the annoying warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        pool = Pool(min((params['experiment']['n_chunks'], cpu_count())))
    else:
        pool = None

    # We import all the libraries with the proper CUDA configuration
    import lib_predict_simple as lib
    importlib.reload(lib)

    if args.external_cat is not None:
        external_cat = pd.read_csv(args.external_cat)
    else:
        external_cat = args.external_cat
    if args.dtypes_table is not None:
        dtypes = pl.read_csv(args.dtypes_table, n_rows = 100).dtypes

    lib.create_dir(args.output_directory)

    total_n_files = len(data_files)

    print('')
    print('We are going to process a total of %s files'%total_n_files)

    start_time = time.time()
    current_nsources = 1
    file_number = 1
    for file_name in data_files:
        print('Processing', file_name)
        output_file = args.output_directory+'/%s_predict_reduced.csv'%(os.path.basename(file_name).split('.')[0])
        if args.incremental and os.path.exists(output_file):
            print('%s file found. Moving to the next one.'%output_file)
            total_n_files -= 1
            continue
        else:
            if os.path.exists(file_name):
                file_number += 1
                print('')
                print('--------------------------------')
                params['variables_n_preproc']['test_catalog'] = file_name
                print('Predicting in %s...'%params['variables_n_preproc']['test_catalog'])
                nsources = model_prediction(params, pool, args.write_test_output, args.output_directory, lib, dtypes = dtypes, external_cat = external_cat)
                current_nsources += nsources

        current_time = time.time()
        print('')
        print('BANNJOS is currently working at %.3f secs per source'%((current_time - start_time)/current_nsources))
        remaining_time = remaining_time_calculator(start_time, current_time, file_number, total_n_files)
        print(remaining_time)

    print('Total elapsed time: %s'%str(timedelta(seconds=(current_time-start_time))))


def model_prediction(params, pool, write_test_output, output_directory, lib, dtypes = None, external_cat = None):

    # Save the results in these files
    output_file = '%s_predict.csv'%(os.path.basename(params['variables_n_preproc']['test_catalog']).split('.')[0])
    output_file_red = '%s_predict_reduced.csv'%(os.path.basename(params['variables_n_preproc']['test_catalog']).split('.')[0])

    # We define the float precision:
    if params['model_nominal']['full_cov']:
        #float_precision = '%.6f'
        float_precision = 6
    else:
        #float_precision = '%.4f'
        float_precision = 4

    test_X, test_X_error, test_output = lib.get_data_test(params['variables_n_preproc'], dtypes = dtypes, external_cat = external_cat)

    test_X_best = test_X.loc[:, params['variables_n_preproc']['best_features']]

    best_features_indexes = test_X_best.columns.get_indexer(params['variables_n_preproc']['best_features'])

    test_X_best_error = test_X_error.iloc[:, best_features_indexes]

    # Indices to save PDFs
    params['variables_n_preproc']['saved_pdfs_indexes'] = np.linspace(0, len(test_X)-1, 10, dtype = int)

    print('Predict data\n')
    predictions_test, y_pred_test = lib.predict(params['model_nominal'], test_X_best, X_error = test_X_best_error, idxs_pred = params['variables_n_preproc']['saved_pdfs_indexes'], n_chunks = params['experiment']['n_chunks'], pool = pool)

    predictions_test.columns = predictions_test.columns.str.replace('pred_nom_|pred_var_', '')

    if params['experiment']['fit_uncertainty_cv']:
        print('Predict data uncertainties\n')
        predictions_test_sigma, y_pred_test_sigma = lib.predict(params['model_variance'], test_X_best, X_error = None, n_chunks = params['experiment']['n_chunks'], pool = pool)

        if write_test_output:
            #test_output.join(predictions_test).join(predictions_test_sigma).to_csv(output_directory+output_file, index=False, float_format=float_precision)
            pl.from_pandas(test_output.join(predictions_test).join(predictions_test_sigma)).write_csv(output_directory+output_file, float_precision=float_precision)

        #test_output.loc[:, ['tile_id', 'number']].join(predictions_test).join(predictions_test_sigma).to_csv(output_directory+output_file_red, index=False, float_format=float_precision)
        pl.from_pandas(test_output.loc[:, ['tile_id', 'number']].join(predictions_test).join(predictions_test_sigma)).write_csv(output_directory+output_file_red, float_precision=float_precision)
    else:
        try:
            class_pc50_col = [col for col in predictions_test.columns.to_list() if ((('CLASS' in col) or ('class' in col)) and ('pc50' in col))]
            class_names = [col.replace('CLASS_', '').replace('class_', '') for col in params['targets_names'] if (('CLASS' in col) or ('class' in col))]
            # CLASS:
            predicted_class = np.argmax(predictions_test.loc[:, class_pc50_col].values, axis = 1)
            predictions_test['CLASS'] = [class_names[i] for i in predicted_class]
        except:
            pass

        if write_test_output:
            #test_output.join(predictions_test).to_csv(output_directory+output_file, index=False, float_format=float_precision)
            pl.from_pandas(test_output.join(predictions_test)).write_csv(output_directory+output_file, float_precision=float_precision)

        #test_output.loc[:, ['tile_id', 'number']].join(predictions_test).to_csv(output_directory+output_file_red, index=False, float_format=float_precision)
        pl.from_pandas(test_output.loc[:, ['tile_id', 'number']].join(predictions_test)).write_csv(output_directory+output_file_red, float_precision=float_precision)

    return len(test_output)

if __name__ == '__main__':

    J_predict(sys.argv[1:])
    sys.exit(0)

"""
Andres del Pino Molina
"""
