import sys, os

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import codecs, json

from multiprocessing import Pool, cpu_count

import glob
import importlib

def model_prediction(params, pool, write_test_output, output_directory, external_cat = None):

    # Save the results in these files
    output_file = '%s_predict.csv'%(os.path.basename(params['variables_n_preproc']['test_catalog']).split('.')[0])
    output_file_red = '%s_predict_reduced.csv'%(os.path.basename(params['variables_n_preproc']['test_catalog']).split('.')[0])

    test_X, test_X_error, test_output = lib.get_data_test(params['variables_n_preproc'], external_cat = external_cat)

    test_X_best = test_X.loc[:, params['variables_n_preproc']['best_features']]

    best_features_indexes = test_X_best.columns.get_indexer(params['variables_n_preproc']['best_features'])

    test_X_best_error = test_X_error.iloc[:, best_features_indexes]

    # Indices to save PDFs
    params['variables_n_preproc']['saved_pdfs_indexes'] = np.linspace(0, len(test_X)-1, 10, dtype = int)

    print('\n')
    print('Predict test data\n')
    predictions_test, y_pred_test = lib.predict(params['model_nominal'], test_X_best, X_error = test_X_best_error, idxs_pred = params['variables_n_preproc']['saved_pdfs_indexes'], n_chunks = params['experiment']['n_chunks'], pool = pool)

    predictions_test.columns = predictions_test.columns.str.replace('pred_nom_|pred_var_', '')

    if params['experiment']['fit_uncertainty_cv']:
        print('\n')
        print('Predict test data uncertainties\n')
        predictions_test_sigma, y_pred_test_sigma = lib.predict(params['model_variance'], test_X_best, X_error = None, n_chunks = params['experiment']['n_chunks'], pool = pool)

        if write_test_output:
            test_output.join(predictions_test).join(predictions_test_sigma.clip(lower=1e-6)).to_csv(output_directory+'/output_prediction/%s'%output_file, index=False, float_format='%.6f')

        test_output.loc[:, ['tile_id', 'number']].join(predictions_test).join(predictions_test_sigma.clip(lower=1e-6)).to_csv(output_directory+'/output_prediction/%s'%output_file_red, index=False, float_format='%.6f')
    else:
        try:
            # CLASS NORMALIZATION
            class_col = [col for col in predictions_test.columns.to_list() if (('CLASS' in col) or ('class' in col) or ('SUBCLASS' in col) or ('subclass' not in col) )]
            predictions_test.loc[:, class_col] = predictions_test.loc[:, class_col].clip(lower=0, upper=1)

            class_pc50_col = [col for col in predictions_test.columns.to_list() if ((('CLASS' in col) or ('class' in col)) and ('pc50' in col))]
            class_names = [col.replace('CLASS_', '').replace('class_', '') for col in params['targets_names'] if (('CLASS' in col) or ('class' in col))]

            predictions_test.loc[:, class_pc50_col] = predictions_test.loc[:, class_pc50_col]/np.expand_dims(predictions_test.loc[:, class_pc50_col].sum(axis = 1).values, axis = 1)

            # CLASS:
            predicted_class = np.argmax(predictions_test.loc[:, class_pc50_col].values, axis = 1)
            predictions_test['CLASS'] = [class_names[i] for i in predicted_class]
        except:
            pass

        if write_test_output:
            test_output.join(predictions_test).to_csv(output_directory+'/output_prediction/%s'%output_file, index=False, float_format='%.6f')

        test_output.loc[:, ['tile_id', 'number']].join(predictions_test).to_csv(output_directory+'/output_prediction/%s'%output_file_red, index=False, float_format='%.6f')


if __name__ == '__main__':

    experiment_name = 'dropout_mean_squared_error_batch_256_layers_700_1300_500_300_dropin_0_drop_0.2_ilr_1e-04_decay_30_PM'

    multiprocesing = True
    divide_data_in_chunks = 12
    alleatoric_n_iter = 300
    write_test_output = False
    external_cat = '/home/adpm/Work/J-PLUS/J-Pipe/CatWISE2020/CatWISE_15.csv'
    data_files = glob.glob('/media/adpm/Extreme Pro/GetJ-Plus/output/J-PLUS_DR3_[0-9]*.csv')

    Interesting_Tiles = ['/media/adpm/Extreme Pro/GetJ-Plus/output/J-PLUS_DR3_90142.csv',
                         '/media/adpm/Extreme Pro/GetJ-Plus/output/J-PLUS_DR3_85369.csv',
                         '/media/adpm/Extreme Pro/GetJ-Plus/output/J-PLUS_DR3_91438.csv',
                         '/media/adpm/Extreme Pro/GetJ-Plus/output/J-PLUS_DR3_98363.csv',
                         '/media/adpm/Extreme Pro/GetJ-Plus/output/J-PLUS_DR3_94193.csv',
                         '/media/adpm/Extreme Pro/GetJ-Plus/output/J-PLUS_DR3_91798.csv']

    if Interesting_Tiles is not None:
        data_files = Interesting_Tiles

    output_directory = '/media/adpm/Extreme Pro/BANNJOS_CatWISE/'

    with open('./results/%s/parameters.json'%experiment_name) as json_file:
        params = json.load(json_file)

    # If you want to change the presictions parameters, it's now or never!
    params['model_nominal']['targets_names'] = [('%s'%name).replace('pred_nom_', '') for name in params['model_nominal']['targets_names']]
    params['experiment']['multiprocesing'] = multiprocesing
    params['experiment']['n_chunks'] = divide_data_in_chunks
    params['model_nominal']['alleatoric_n_iter'] = alleatoric_n_iter

    if params['experiment']['multiprocesing']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        pool = Pool(min((params['experiment']['n_chunks'], cpu_count())))
    else:
        pool = None

    # We import all the libraries with the proper CUDA configuration
    import lib_predict_simple as lib
    importlib.reload(lib)

    if external_cat is not None:
        external_cat = pd.read_csv(external_cat)

    lib.create_dir(output_directory+'/output_prediction/')

    for file_name in data_files:
        print(file_name)
        if os.path.exists(file_name):
            params['variables_n_preproc']['test_catalog'] = file_name
            print('Predicting in %s...'%params['variables_n_preproc']['test_catalog'])
            model_prediction(params, pool, write_test_output, output_directory, external_cat = external_cat)

    sys.exit()
