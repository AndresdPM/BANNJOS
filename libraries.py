import sys, os

from multiprocessing import Pool, cpu_count

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import polars as pl

import urllib.request
from math import log10, floor
import time
import datetime as dt
import uncertainties.unumpy as unp

import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rc('text', usetex=True)

def RFE_selector(X, y, params, scoring = 'accuracy', min_features_to_select = 5, n_splits=5, step = 1, output_name = 'Recursive_feature_selection'):

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFECV

    forest = RandomForestRegressor(n_estimators=32, n_jobs = -1, verbose = 0)

    rfecv = RFECV(
        estimator=forest,
        step=step,
        cv=n_splits,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        verbose = 1)
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.close('all')

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (mae)")
    plt.plot(range(min_features_to_select, (len(rfecv.cv_results_['mean_test_score']) * step) + min_features_to_select, step),
             rfecv.cv_results_['mean_test_score'], zorder = 1
             )

    plt.fill_between(range(min_features_to_select, (len(rfecv.cv_results_['mean_test_score']) * step) + min_features_to_select, step), rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'], rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'], alpha = 0.3, zorder = 0)

    plt.savefig(output_name+'.png')

    rfe_values = rfecv.get_support()
    best_features = X.loc[:, rfe_values].columns.to_list()

    pd.Series(name = '#best_features', data=best_features).to_csv(output_name+'_bf.csv', index = False)
    pd.DataFrame(rfecv.cv_results_).to_csv(output_name+'_cv.csv', index = False)
    pd.DataFrame(columns = X.columns, data = np.expand_dims(rfecv.ranking_, axis=0)).to_csv(output_name+'_ranking.csv', index = False)

    return best_features, rfecv.ranking_ , rfecv.cv_results_


def create_dir(path):
   """
   This routine creates directories.
   """

   if not os.path.isdir(path):
      try:
         tree = path.split('/')
         previous_tree = tree[0]
         for leave in tree[1:]:
            previous_tree = '%s/%s'%(previous_tree,leave)
            try:
               os.mkdir(previous_tree)
            except:
               pass
      except OSError:
         print ("Creation of the directory %s failed" % path)
      else:
         print ("Successfully created the directory %s " % path)


def dict_merge(dct, merge_dct):
    """
    Recursive dict merge
    """
    from collections.abc import Mapping

    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def NLL(y_true, y_pred):
  return -y_pred.log_prob(y_true)


def normal_sp(params):
    shape = params.shape[1]
    return tfp.distributions.Normal(loc=params[:, 0:int(shape/2)], scale=1e-4 + tf.math.softplus(0.05 * params[:,int(shape/2):int(shape)])) # both parameters are learnable


def normal_exp(params):
    shape = params.shape[1]
    return tfp.distributions.Normal(loc=params[:,0:int(shape/2)], scale=tf.math.exp(params[:, int(shape/2):int(shape)])) # both parameters are learnable


def random_forest(n_inputs, n_outputs, n_instances, params):

    model = RandomForestRegressor(n_estimators=params['n_estimators'], n_jobs = -1, verbose = params['verbose'])

    return model


def k_neighbors(n_inputs, n_outputs, n_instances, params):

    model = KNeighborsRegressor(n_neighbors=params['n_neighbors'], weights=params['weights'], p=params['power_minkowski'], n_jobs = 1)

    return model


def gaussian_posterior(n_inputs, n_outputs, n_instances, params):

    learning_rate = float(params['initial_learning_rate'])

    inputs = Input(shape=(n_inputs,), name = 'input')

    for layer in range(params['hidden_layers']):
        if layer == 0:
            previous = inputs
        else:
            previous = hidden

        hidden = tf.keras.layers.Dense(params['hidden_neurons_%i'%(layer+1)], activation="relu", name = 'hidden_%i'%layer)(previous)

    params_mc = Dense(n_outputs * 2)(hidden)
    dist = tfp.layers.DistributionLambda(normal_sp)(params_mc)

    model_nobay = Model(inputs=inputs, outputs=dist, name='Gauss')
    model_nobay.compile(Adam(learning_rate=learning_rate), loss=params['loss'])

    return model_nobay


def deterministic(n_inputs, n_outputs, n_instances, params):

    learning_rate = float(params['initial_learning_rate'])
    inputs = Input(shape=(n_inputs,), name = 'input')

    for layer in range(params['hidden_layers']):
        if layer == 0:
            previous = inputs
        else:
            previous = hidden
        hidden = Dense(params['hidden_neurons_%i'%(layer+1)], activation="relu", name = 'hidden_%i'%layer)(previous)

    output = Dense(n_outputs)(hidden)

    model = Model(inputs=inputs, outputs=output, name='Deterministic')
    model.compile(Adam(learning_rate=learning_rate), loss=params['loss'])

    return model


def variational_inference(n_inputs, n_outputs, n_instances, params):

    learning_rate = float(params['initial_learning_rate'])

    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_instances * 1.0)
    bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (n_instances * 1.0)

    inputs = Input(shape=(n_inputs,), name = 'input')

    for layer in range(params['hidden_layers']):

        if layer == 0:
            previous = inputs
        else:
            previous = hidden

        hidden = tfp.layers.DenseFlipout(params['hidden_neurons_%i'%(layer+1)],
                                         bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                         kernel_divergence_fn=kernel_divergence_fn,
                                         bias_divergence_fn=bias_divergence_fn,
                                         activation="relu", name = 'hidden_%i'%layer)(previous)

    params_mc = tfp.layers.DenseFlipout(n_outputs * 2,
                                     bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                     kernel_divergence_fn=kernel_divergence_fn,
                                     bias_divergence_fn=bias_divergence_fn)(hidden)

    dist = tfp.layers.DistributionLambda(normal_sp)(params_mc)

    model_vi = Model(inputs=inputs, outputs=dist, name='Variational_Inference')
    model_vi.compile(Adam(learning_rate=learning_rate), loss=params['loss'])

    return model_vi


def dropout(n_inputs, n_outputs, n_instances, params):

    learning_rate = float(params['initial_learning_rate'])

    inputs = Input(shape=(n_inputs,))
    if params['input_dropout'] > 0:
        previous = Dropout(params['input_dropout'])(inputs, training=True)
    else:
        previous = inputs

    for layer, (hidden_neurons, hidden_dropout) in enumerate(zip(params['hidden_layers'], params['hidden_dropout'])):
        previous = Dense(hidden_neurons, activation="relu")(previous)
        previous = Dropout(hidden_dropout)(previous, training=True)

    params_mc = Dense(n_outputs * 2)(previous)
    dist_mc = tfp.layers.DistributionLambda(normal_sp, name='normal_sp')(params_mc)

    model_mc = Model(inputs=inputs, outputs=dist_mc, name='Dropout')
    model_mc.compile(Adam(learning_rate=learning_rate), loss=params['loss'])

    return model_mc


def run_experiment_cv(params, X, y, X_error= None, y_error = None, n_splits=10, n_repeats=10, stratify = None, n_chunks = 1, pool = None):

    def exp_decay(epoch):
        initial_lrate = 0.1
        k = 0.1
        lrate = initial_lrate * np.exp(-k*t)
        sys.stdout.write("\rEpoch {0}, using learning rate of {1}%".format(epoch, lrate))
        sys.stdout.flush()
        return lrate

    def step_decay(epoch, lr):
        if ((epoch+1) % params['step_decay_learning_rate'] == 0):
            lr = round(lr * 0.5, -int(floor(log10(abs(lr * 0.5)))) + 1)
            print('LR =', max(params['final_learning_rate'], lr))
        return max(params['final_learning_rate'], lr)

    def calcProcessTime(starttime, cur_iter, max_iter):

        telapsed = time.time() - starttime
        testimated = (telapsed/cur_iter)*(max_iter)

        finishtime = starttime + testimated
        finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

        lefttime = testimated-telapsed  # in seconds

        return (int(telapsed)/60., int(lefttime)/60., finishtime)

    features_names, targets_names = X.columns.to_list(), y.columns.to_list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]

    if 'train' in params['mode']:

        epochs = int(params['epochs'])
        batch_size = int(params['batch_size'])

        lrs = tf.keras.callbacks.LearningRateScheduler(step_decay)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = params['delta_early_stop_patience'], patience=params['early_stop_patience'], restore_best_weights=True)

        if ((params['step_decay_learning_rate'] > 0) & (params['step_decay_learning_rate'] < epochs)):
            callbacks = [es, lrs]
        else:
            callbacks = [es]

    model = globals()[params['model']]

    if stratify is not None:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
        y_cv = stratify
    else:
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
        y_cv = y

    losses = []
    val_losses = []
    results = []
    valid_indices = []
    for ii, (train_ind, valid_ind) in enumerate(cv.split(X, y_cv)):
        print('\nCross validation', ii)

        tf.keras.backend.clear_session()

        X_train, X_valid = X.iloc[train_ind,:], X.iloc[valid_ind,:]
        y_train, y_valid = y.iloc[train_ind,:], y.iloc[valid_ind,:]

        try:
            X_error_train, X_error_valid = X_error.iloc[train_ind,:], X_error.iloc[valid_ind,:]
            y_error_train, y_error_valid = y_error.iloc[train_ind,:], y_error.iloc[valid_ind,:]
        except:
            X_error_valid = None
            y_error_valid = None

        # We save the indices
        valid_indices.append(valid_ind)

        n_instances = X_train.shape[0]
        built_model = model(n_inputs, n_outputs, n_instances, params)

        if ii == 0:
            print(built_model.summary())
            print(callbacks)

        history = built_model.fit(X_train.values, y_train.values, epochs=epochs, batch_size=batch_size, verbose=params['verbose'], callbacks=callbacks, validation_split=0.3)

        losses.append(pd.DataFrame(data={'loss_%i'%ii:history.history['loss'], 'val_loss_%i'%ii:history.history['val_loss']}))

        results.append(predict_validate(params, X_valid, y_valid, X_valid_error = X_error_valid, y_valid_error = y_error_valid, built_model = built_model, idxs_pred = None, n_chunks = n_chunks, pool = pool)[0])

    # Take the average probabilty on 5 folds
    results = pd.concat(results)
    losses = pd.concat(losses, axis = 1)
    valid_indices = [item for sublist in valid_indices for item in sublist]

    tf.keras.backend.clear_session()
    del [built_model]

    return valid_indices, results, losses


def get_statistics(preds, targets_names, full_cov, pool = None, index = None):

    from astropy.stats import sigma_clip, sigma_clipped_stats, mad_std
    from sklearn.mixture import GaussianMixture

    # The format of preds is (n_iterations, n_objects, n_features)
    clipped_pred = np.ma.filled(sigma_clip(preds, axis = 0, sigma = 6, stdfunc='mad_std', maxiters=10), np.nan)

    # Standard statistics
    pc = np.nanpercentile(clipped_pred, [2.275, 15.865, 50.0, 84.135, 97.725], axis=0)
    std = mad_std(clipped_pred, axis = 0, ignore_nan=True)
    mean = np.nanmean(clipped_pred, axis = 0)

    # The correlations have to be obtained in a loop
    triu_indices = np.triu_indices(len(targets_names), k = 1)
    correlation_names = [targets_names[ii]+'_'+targets_names[jj]+'_corr' for (ii, jj) in zip(triu_indices[0], triu_indices[1])]

    correlation_tf = tfp.stats.correlation(preds, sample_axis=0, event_axis=-1, keepdims=False, name=None).numpy()
    correlation = np.ones((preds.shape[1], len(correlation_names)))*-99
    for ii, corr in enumerate(correlation_tf):
        cli_progress_test(ii+1, preds.shape[1])
        correlation[ii, :] = corr[triu_indices]
    del [correlation_tf]

    results = pd.DataFrame(columns=[target+'_mean' for target in targets_names]+[target+'_std' for target in targets_names]+correlation_names+[target+'_pc02' for target in targets_names]+[target+'_pc16' for target in targets_names]+[target+'_pc50' for target in targets_names]+[target+'_pc84' for target in targets_names]+[target+'_pc98' for target in targets_names], data = np.hstack([mean, std, correlation, pc[0], pc[1], pc[2], pc[3], pc[4]]), index = index)

    if full_cov:
        print('')
        print('Fitting the GMM...')
        # Number of components for the GMM
        gmm_n_comp = len(targets_names)

        triu_indices = np.triu_indices(preds.shape[2]-1, k = 0)
        rot_covariances_names = ['comp%i_cov_%i%i'%(comp+1, ii+1, jj+1) for comp in range(gmm_n_comp) for (ii, jj) in zip(triu_indices[0], triu_indices[1])]
        rot_means_names = ['comp%i_mean_%i'%(comp+1, ii+1) for comp in range(gmm_n_comp) for ii in range(preds.shape[2]-1)]
        rot_weights_names = ['comp%i_weight'%(comp+1) for comp in range(gmm_n_comp)]

        rot_covariances = np.ones((preds.shape[1], len(rot_covariances_names)))*-99
        rot_means = np.ones((preds.shape[1], len(rot_means_names)))*-99
        rot_weights = np.ones((preds.shape[1], len(rot_weights_names)))*-99

        # We perform the entire GMM compression
        mat = np.array([[(np.sqrt(3)+3)/6, -np.sqrt((2-np.sqrt(3))/6), -1/np.sqrt(3)],
        [-np.sqrt((2-np.sqrt(3))/6), (np.sqrt(3)+3)/6, -1/np.sqrt(3)],
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])

        rot_preds = np.dot(preds, mat.T)

        gmm = GaussianMixture(n_components=gmm_n_comp, init_params = 'k-means++')

        # Parallelize GMM
        args_gmm = []
        for pred_chunk, index_chunk in zip(np.array_split(rot_preds.swapaxes(0,1), pool._processes), np.array_split(index, pool._processes)):
            args_gmm.append((gmm, pred_chunk, index_chunk))

        results = results.join(pd.concat(pool.map(launch_gmm_predictions, args_gmm)))

    return results


def launch_gmm_predictions(args):
    """
    This routine pipes into multiple threads.
    """
    return gmm_prediction(*args)


def gmm_prediction(gmm, data, index):
    """
    This routine fits a multi gaussian mixture model into data and returns the parameters of such fit
    """
    gmm_n_comp = gmm.n_components
    triu_indices = np.triu_indices(data.shape[2]-1, k = 0)

    covariances_names = ['comp%i_cov_%i%i'%(comp+1, ii+1, jj+1) for comp in range(gmm_n_comp) for (ii, jj) in zip(triu_indices[0], triu_indices[1])]
    means_names = ['comp%i_mean_%i'%(comp+1, ii+1) for comp in range(gmm_n_comp) for ii in range(data.shape[2]-1)]
    weights_names = ['comp%i_weight'%(comp+1) for comp in range(gmm_n_comp)]

    covariances = np.ones((data.shape[0], len(covariances_names)))*-99
    means = np.ones((data.shape[0], len(means_names)))*-99
    weights = np.ones((data.shape[0], len(weights_names)))*-99

    for ii, obj in enumerate(data):
        cli_progress_test(ii+1, data.shape[0])
        gmm.fit(obj[:,0:2])
        covariances[ii,:] = np.concatenate([cov[triu_indices] for cov in gmm.covariances_])
        means[ii,:] = gmm.means_.flatten()
        weights[ii,:] = gmm.weights_.flatten()

    results = pd.DataFrame(columns=covariances_names+means_names+weights_names, data = np.hstack([covariances, means, weights]), index = index)

    return results


def cli_progress_test(current, end_val, bar_length=50):
    """
    Just a progress bar
    """
    percent = float(current) / end_val
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rProcessing: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()


def build_model(params, X_train = None, y_train = None, weights_file = None):

    def exp_decay(epoch):
        initial_lrate = 0.1
        k = 0.1
        lrate = initial_lrate * np.exp(-k*t)
        sys.stdout.write("\rEpoch {0}, using learning rate of {1}%".format(epoch, lrate))
        sys.stdout.flush()
        return lrate

    def step_decay(epoch, lr):
        if ((epoch+1) % params['step_decay_learning_rate'] == 0):
            lr = round(lr * 0.5, -int(floor(log10(abs(lr * 0.5)))) + 1)
            print('LR =', max(params['final_learning_rate'], lr))
        return max(params['final_learning_rate'], lr)

    def variable_step_decay(epoch, lr):
        if ((epoch+1) % params['step_decay_learning_rate'] == 0):
            lr = round(lr * 0.5, -int(floor(log10(abs(lr * 0.5)))) + 1)
            params['step_decay_learning_rate'] *= 2   # This line can be removed if it fails.
            print('LR =', max(params['final_learning_rate'], lr))
        return max(params['final_learning_rate'], lr)


    print('Running with params:', params)

    model = globals()[params['model']]

    if 'train' in params['mode']:

        tf.keras.backend.clear_session()

        n_inputs, n_instances, n_outputs = X_train.shape[1], X_train.shape[0], y_train.shape[1]
        built_model = model(n_inputs, n_outputs, n_instances, params)

        epochs = int(params['epochs'])
        batch_size = int(params['batch_size'])

        print(built_model.summary())
        print('delta_early_stop_patience = %s'%params['delta_early_stop_patience'])

        lrs = tf.keras.callbacks.LearningRateScheduler(step_decay)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = params['delta_early_stop_patience'], patience=params['early_stop_patience'], restore_best_weights=True)

        if params['step_decay_learning_rate']:
            callbacks = [es, lrs]
        else:
            callbacks = [es]

        history = built_model.fit(X_train.values, y_train, epochs=epochs, batch_size=batch_size, verbose=params['verbose'], callbacks=callbacks, validation_split=0.3, use_multiprocessing = True, workers = 16)
        losses = pd.DataFrame(data={'loss':history.history['loss'], 'val_loss':history.history['val_loss']})

    if 'predict' in params['mode']:
        built_model = model(params['pre_trained_n_inputs'], params['pre_trained_n_outputs'], params['pre_trained_n_instances'], params)

        print(built_model.summary())

        built_model.load_weights(params['pre_trained_weights_file'])
        losses = pd.DataFrame()

    if 'just_fit' in params['mode']:

        built_model = built_model.fit(X_train.values, y_train)
        losses = pd.DataFrame()

    return built_model, losses


def predict_validate(params, X_valid, y_valid, X_valid_error = None, y_valid_error = None, built_model = None, idxs_pred = None, n_chunks = 1, pool = None):

    if y_valid_error is None:
        y_valid_error = pd.DataFrame(data=np.zeros_like(y_valid), columns=['%s_error'%col for col in y_valid.columns.to_list()], index = y_valid.index)

    if X_valid_error is None:
        X_valid_error = pd.DataFrame(data=np.zeros_like(X_valid), columns=['%s_error'%col for col in X_valid.columns.to_list()], index = X_valid.index)

    results, y_pred_sel = predict(params, X_valid, X_error = X_valid_error, built_model = built_model, idxs_pred = idxs_pred, n_chunks = n_chunks, pool = pool)

    results = y_valid.join(y_valid_error).join(results)

    return results, y_pred_sel


def predict(params, X, X_error = None, built_model = None, idxs_pred = None, n_chunks = 1, pool = None):

    if X_error is None:
        X_error = pd.DataFrame(data=np.zeros_like(X), columns=['%s_error'%col for col in X.columns.to_list()], index = X.index)

    if n_chunks > 1:
        if pool is not None:
            print('Running prediction in multiprocessing mode.')
            args_prediction = []
            for X_i, X_error_i in zip(np.array_split(X, n_chunks), np.array_split(X_error, n_chunks)):
                args_prediction.append((params, X_i, X_error_i, built_model))

            y_pred = np.concatenate(pool.map(launch_make_predictions, args_prediction), axis = 1)

            results = get_statistics(y_pred, params['targets_names'], params['full_cov'], pool = pool, index = X.index)

            if idxs_pred is not None:
                y_pred_sel = y_pred[:, [idx for idx in idxs_pred if idx < np.shape(y_pred)[1]], :]

            del [y_pred]

        else:
            print('Running prediction in memory saving mode.')
            y_pred_sel = []
            results = []
            for ii, (indices_total, X_i, X_error_i) in enumerate(zip(np.array_split(range(X.shape[0]), n_chunks), np.array_split(X, n_chunks), np.array_split(X_error, n_chunks))):
                print('\n')
                print('Predicting %i of %i'%(ii+1, n_chunks))

                y_pred_i = make_predictions(params, X_i, X_error_i, built_model)

                results.append(get_statistics(y_pred_i, params['targets_names'], params['full_cov'], pool = Pool(int(cpu_count()/2)), index = X_i.index))

                if idxs_pred is not None:
                    indices = np.where(np.in1d(indices_total, [idx for idx in idxs_pred if idx < np.shape(X)[0]]))[0]
                    y_pred_sel.append(np.take(y_pred_i, indices, axis=1))

                del [y_pred_i]

            results = pd.concat(results)

            if idxs_pred is not None:
                y_pred_sel = np.concatenate(y_pred_sel, axis =1)

    else:
        print('Running prediction single processor mode.')
        y_pred = make_predictions(params, X, X_error, built_model)
        results = get_statistics(y_pred, params['targets_names'], params['full_cov'], pool = Pool(int(cpu_count()/2)), index = X.index)
        y_pred_sel = y_pred[:, [idx for idx in idxs_pred if idx < np.shape(y_pred)[1]], :]

        del [y_pred]

    return results, y_pred_sel


def launch_make_predictions(args):
    """
    This routine pipes gaia_query into multiple threads.
    """
    return make_predictions(*args)


def make_predictions(params, X, eX = None, built_model = None, process = None):

    from numpy.random import default_rng
    tf.keras.backend.clear_session()

    if built_model is None:

        model = globals()[params['model']]

        built_model = model(params['pre_trained_n_inputs'], params['pre_trained_n_outputs'], params['pre_trained_n_instances'], params)

        if process is not None:
            model_n = '_%s'%process
        else:
            model_n = ''

        built_model.load_weights(params['pre_trained_weights_file'])

    rng_val = default_rng(process)

    iterate = int(params['alleatoric_n_iter'])
    preds =np.zeros((iterate, X.shape[0], params['pre_trained_n_outputs']))

    if params['alleatoric_montecarlo']:
        for kk in range(0, iterate):
            cli_progress_test(kk+1, iterate)
            preds[kk, :, :] = built_model.predict(rng_val.normal(X, eX), verbose=0)
    else:
        for kk in range(0, iterate):
            cli_progress_test(kk+1, iterate)
            preds[kk, :, :] = built_model.predict(X, verbose=0)

    return preds


def get_data(params):

    def rel2abs(var, var_relerr):
        return pd.DataFrame(data = var_relerr.values * var.values, columns = ['%s_error'%col.replace('relerr_', '').replace('_relerr', '').replace('relerr', '') for col in var_relerr.columns], index = var_relerr.index)


    def get_errors(data, used_cols = None, force_cols = None):

        if force_cols is not None:
            errors = data.loc[:, force_cols]

        else:
            errors = pd.DataFrame(index = data.index)

            all_cols = data.columns

            if not used_cols:
                used_cols = [x for x in cols if (not '_error' in x) & (not 'err' in x) ]

            for used_col in used_cols:
                if '%s_error'%used_col in all_cols:
                    errors['%s_error'%used_col] = data['%s_error'%used_col]
                elif len(used_col.split('mag_')) > 1:
                    if 'mag_err_%s'%used_col.split('mag_')[1] in all_cols:
                        errors['mag_err_%s'%used_col.split('mag_')[1]] = data['mag_err_%s'%used_col.split('mag_')[1]]
                    else:
                        errors['%s_error'%used_col] = np.nan
                elif '%s_err'%used_col in all_cols:
                    errors['%s_err'%used_col] = data['%s_err'%used_col]
                elif '%s_ERR'%used_col in all_cols:
                    errors['%s_ERR'%used_col] = data['%s_ERR'%used_col]
                elif '%s_RMS'%used_col in all_cols:
                    errors['%s_RMS'%used_col] = data['%s_RMS'%used_col]
                elif ('%s_ERRZPT'%used_col.replace('_ZPT', '') in all_cols):
                    errors['%s_ERRZPT'%used_col.replace('_ZPT', '')] = data['%s_ERRZPT'%used_col.replace('_ZPT', '')]
                elif 'e_%s'%used_col in all_cols:
                    errors['e_%s'%used_col] = data['e_%s'%used_col]
                else:
                    errors['%s_error'%used_col] = np.nan

        return errors, errors.columns.to_list()


    def get_photometry(data, used_photo):
        # Wich columns could be interesting? Lets start with the WORST PSF
        if used_photo=='flux_psfcor':
            photo = [mag for mag in data.columns if 'flux_psfcor' in mag]
            photo_err = [flux for flux in data.columns if 'flux_relerr_psfcor' in flux]
        if used_photo=='corr_photo_3':
            photo = [mag for mag in data.columns if 'corr_mag_aper_3_0' in mag]
            photo_err = [flux for flux in data.columns if 'flux_relerr_aper_3' in flux]
        if used_photo=='photo_3':
            photo = [mag for mag in data.columns if (('mag_aper_3_0' in mag) & ('corr_' not in mag))]
            photo_err = [flux for flux in data.columns if 'flux_relerr_aper_3' in flux]
        if used_photo=='flux_3_worstpsf':
            photo = [mag for mag in data.columns if 'flux_aper3_worstpsf_' in mag]
            photo_err = [flux for flux in data.columns if 'flux_relerr_aper3_worstpsf' in flux]
        if used_photo=='mag_3_worstpsf':
            photo = [mag for mag in data.columns if 'mag_aper3_worstpsf_' in mag]
            photo_err = [flux for flux in data.columns if 'flux_relerr_aper3_worstpsf' in flux]
        if used_photo=='flux_aper_3_0':
            photo = [mag for mag in data.columns if 'flux_aper_3_0_' in mag]
            photo_err = [flux for flux in data.columns if 'flux_relerr_aper_3_0_' in flux]
        if used_photo=='fnu_flux_aper':
            photo = [mag for mag in data.columns if 'fnu_flux_aper_' in mag]
            photo_err = [flux for flux in data.columns if 'fnu_flux_relerr_aper_' in flux]
        if used_photo=='flux_aper':
            photo = [mag for mag in data.columns if (('flux_aper_' in mag) and ('_flux_aper_' not in mag)) ]
            photo_err = [flux for flux in data.columns if (('flux_relerr_aper_' in flux) and ('_flux_relerr_aper_' not in flux))]
        if used_photo=='mag_aper':
            photo = [mag for mag in data.columns if 'mag_aper_' in mag]
            photo_err = [flux for flux in data.columns if 'mag_err_aper_' in flux]

        return photo, photo_err


    def transform_errors(data, photo, photo_err):
        # We transform relative to absolute errors
        abs_errors = rel2abs(data.loc[:, photo], data.loc[:, photo_err])
        data = data.join(abs_errors)
        photo_err = abs_errors.columns.to_list()
        return data, photo_err

    X_vars, y_vars, y_vars_err, validation_vars, quality_vars, prediction_vars = params['X_vars'], params['y_vars'], params['y_vars_err'], params['validation_vars'], params['quality_vars'], params['prediction_vars']

    # Read data
    if params['nrows'] is not None:
        #training_data = pd.read_csv(params['training_catalog'], nrows=params['nrows'])
        training_data = pl.read_csv(params['training_catalog'], nrows=params['nrows']).to_pandas()
    else:
        if (params['skip_nrows'] is not None) and (params['each_nrows'] > 1):
            skip_function = lambda i: (i+params['skip_nrows']) % params['each_nrows'] != 0
        elif (params['each_nrows'] > 1):
            skip_function = lambda i: i % params['each_nrows'] != 0
        elif (params['skip_nrows'] is not None):
            skip_function = params['skip_nrows']
        else:
            skip_function = 0

        #training_data = pd.read_csv(params['training_catalog'], skiprows=skip_function)
        training_data = pl.read_csv(params['training_catalog'], skip_rows=skip_function).to_pandas()

    print('We have read %i lines.'%len(training_data))

    #Shuffle data
    training_data = shuffle(training_data, random_state = params['random_generator_seed'])

    try:
        print('For the moment, the distribution of classes is as follow:\n', training_data.groupby(['CLASS'])['CLASS'].count())
    except:
        pass

    # We may want to select a specific class
    if params['select_class'] is not None:
        if isinstance(params['select_class'], list):
            training_data = training_data.loc[training_data.CLASS.isin(params['select_class'])]
        elif isinstance(params['select_class'], str):
            training_data = training_data.loc[training_data.CLASS == params['select_class']]


    if params['y_drop_nans']:
        # We drop NaN values, if present.
        y_float_vars = training_data.loc[:, y_vars+y_vars_err].select_dtypes(exclude=['object']).columns
        training_data.loc[:, y_float_vars] = training_data.loc[:, y_float_vars].apply(lambda x: np.where(x < -999.0, np.nan, x)).values
        training_data = training_data.loc[training_data.loc[:, y_vars+y_vars_err].notnull().all(axis = 1), :]
    else:
        # We simply assign a label -999.0 to the missing data
        training_data.loc[:, y_vars+y_vars_err] = training_data.loc[:, y_vars+y_vars_err].fillna(-999.0)

    print('After selection based on y_bars we continue with %i lines.'%len(training_data))

    photo, photo_err = get_photometry(training_data, params['used_photo'])

    #first abs values for PMs
    pmra_g = unp.uarray(training_data.pmra_g, training_data.e_pmra_g)
    pmdec_g = unp.uarray(training_data.pmde_g, training_data.e_pmde_g)
    pm_g = unp.sqrt(pmra_g**2 + pmdec_g**2)
    training_data['pm_g_error'] = unp.std_devs(pm_g)
    training_data['pm_g'] = unp.nominal_values(pm_g)
    X_vars = X_vars + ['pm_g_error', 'pm_g']
    X_vars.remove('e_pmra_g')
    X_vars.remove('e_pmde_g')
    X_vars.remove('pmra_g')
    X_vars.remove('pmde_g')
    del [pmra_g, pmdec_g, pm_g]

    pmra_cw  = unp.uarray(training_data.pmra_cw*1000, np.abs(training_data.e_pmra_cw)*1000)
    pmdec_cw = unp.uarray(training_data.pmde_cw*1000, np.abs(training_data.e_pmde_cw)*1000)
    pm_cw = unp.sqrt(pmra_cw**2 + pmdec_cw**2)
    training_data['e_pm_cw'] = unp.std_devs(pm_cw)
    training_data['pm_cw'] = unp.nominal_values(pm_cw)
    X_vars = X_vars + ['e_pm_cw', 'pm_cw']
    X_vars.remove('e_pmra_cw')
    X_vars.remove('e_pmde_cw')
    X_vars.remove('pmra_cw')
    X_vars.remove('pmde_cw')
    del [pmra_cw, pmdec_cw, pm_cw]

    #Use Galactic latitude?
    if params['use_gal_lat']:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        c = SkyCoord(ra=training_data.alpha_j2000.values*u.degree, dec=training_data.delta_j2000.values*u.degree, frame='fk5')
        training_data['l'] = c.galactic.l.value
        training_data['b'] = c.galactic.b.value
        X_vars = X_vars + ['b']

    X_vars = X_vars+photo

    try:
        print('After cleaning, the distribution of classes is as follow:\n', training_data.groupby(['CLASS'])['CLASS'].count())
    except:
        pass

    if params['use_photo_error']:
        X_vars = X_vars + photo_err

    if (params['photo_log']) & ('flux' in params['used_photo']):
        # We transform the fluxes to logarithm
        fluxes_zpt = training_data.loc[:, photo].min().quantile(0.5)
        training_data.loc[:, photo_err] = (training_data.loc[:, photo_err].values)/((training_data.loc[:, photo].values)*np.log(10))
        training_data.loc[:, photo] = np.log10(training_data.loc[:, photo] - fluxes_zpt)
        try:
            training_data.loc[:, [ 'e_fg','e_fbp','e_frp']] = (training_data.loc[:, [ 'e_fg','e_fbp','e_frp']].values)/((training_data.loc[:, [ 'fg','fbp','frp']].values)*np.log(10))
            training_data.loc[:, [ 'fg','fbp','frp']] = np.log10(training_data.loc[:, [ 'fg','fbp','frp']] - fluxes_zpt)
        except:
            pass
    else:
        fluxes_zpt = 0

    # We are going to add new colors that help with QSO detection
    import itertools
    photo_colors = [mag for mag in training_data.columns if 'magab_mag_aper_4' in mag]+['w1mpropm_cw', 'w2mpropm_cw']

    # We transform the missing magnitudes to nans
    training_data.loc[:, photo_colors] = training_data.loc[:, photo_colors].apply(lambda x: np.where(x == 99.0, np.nan, x)).values

    # We try to get the errors for X_vars
    training_eX, eX_vars = get_errors(training_data, used_cols = X_vars)

    # We select the data
    training_output = training_data.loc[:, validation_vars]
    quality_data = training_data.loc[:, quality_vars]
    training_data = training_data.loc[:, X_vars+y_vars+y_vars_err]

    # We make sure that -999 are treated as nans in the X vector
    training_data.loc[:, X_vars] = training_data.loc[:, X_vars].apply(lambda x: np.where(x <= -999, np.nan, x)).values

    # We standarize the data and the errors:
    minimum = training_data.loc[:, X_vars].quantile(0.005, axis=0).values
    maximum = training_data.loc[:, X_vars].quantile(0.995, axis=0).values
    dynamic_range = maximum - minimum
    non_zero_dynamic_range = dynamic_range > 0

    vars_non_zero_dynamic_range = [i for (i, v) in zip(X_vars, non_zero_dynamic_range) if v]
    e_vars_non_zero_dynamic_range = [i for (i, v) in zip(eX_vars, non_zero_dynamic_range) if v]

    training_data.loc[:, vars_non_zero_dynamic_range] = (training_data.loc[:, vars_non_zero_dynamic_range].values - minimum[non_zero_dynamic_range]) / dynamic_range[non_zero_dynamic_range]

    training_eX.loc[:, e_vars_non_zero_dynamic_range] = training_eX.loc[:, e_vars_non_zero_dynamic_range].values / dynamic_range[non_zero_dynamic_range]

    training_data.loc[:, X_vars] = training_data.loc[:, X_vars].clip(lower = 0, upper = 1)
    training_eX.loc[:, eX_vars] = training_eX.loc[:, eX_vars].clip(lower = 0, upper = 1)

    # We obtain all the classes probabilities
    try:
        from sklearn.preprocessing import LabelEncoder
        y_class = training_data.loc[:, y_vars+y_vars_err].select_dtypes(include='object')

        y_class_dummies = pd.get_dummies(y_class)
        y_var_class = y_class_dummies.columns.to_list()
        label_encoder = LabelEncoder()
        for col in y_class.columns:
            y_class_dummies.loc[:, '%s_num'%col] = label_encoder.fit_transform(y_class[col])

        training_data = training_data.join(y_class_dummies)
        y_test_categorical = y_class.columns.to_list() + y_class.add_suffix('_num').columns.to_list()
        y_vars = list(set(y_vars).difference(y_class.columns.to_list())) + y_var_class
    except:
        y_test_categorical = []

    # We convert the nan to a figure just outside the std of the distribution.
    training_data = training_data.fillna(params['fill_na'])

    # If there's no error, then zero
    training_ey, ey_vars = get_errors(training_data, used_cols = y_vars, force_cols = y_vars_err)
    training_eX = training_eX.fillna(0).clip(lower=0)
    training_ey = training_ey.fillna(0).clip(lower=0)

    training_X = training_data.loc[:, X_vars]
    training_y = training_data.loc[:, y_vars]
    training_y_cat = training_data.loc[:, y_test_categorical]

    print('The training independent vector is', training_X.shape)
    print('The training dependent vector is', training_y.shape)

    if params['test_catalog'] is not None:
        predict_data = pd.read_csv(params['test_catalog'])

        predict_output = predict_data.loc[:, prediction_vars]

        # We make sure that 99 the photometric magnitudes are treated as Nans:
        predict_data.loc[:, photo+photo_err] = predict_data.loc[:, photo+photo_err].apply(lambda x: np.where(x == 99.0, np.nan, x)).values

        if (len(photo_err) > 0) & ('flux' in params['used_photo']):
            predict_data = transform_errors(predict_data, photo, photo_err)[0]

        # We select the data
        predict_X = predict_data.loc[:, X_vars]
        predict_eX = get_errors(predict_data, used_cols = X_vars)[0]

        # We make sure that -999 are treated as nans, and that there are not nans in the y
        predict_X = predict_X.apply(lambda x: np.where(x < -999, np.nan, x))

        # We apply the same normalization
        predict_X = (predict_X - minimum) / dynamic_range

        # We convert the nan to a figure just outside the std of the distribution.
        predict_X = predict_X.fillna(params['fill_na'])
        predict_eX = predict_eX.fillna(0).clip(lower=0)

    else:
        predict_X, predict_eX, predict_output = None, None, None

    return training_X, training_eX, training_y, training_ey, training_y_cat, training_output, quality_data, X_vars, y_vars, y_vars_err, predict_X, predict_eX, predict_output, minimum, dynamic_range, fluxes_zpt


def get_train_val(X, eX, y, ey, y_cat, output, params):

    if params['random_generator_seed'] is None:
        import random
        random_state = random.randrange(1e3)
    else:
        random_state = params['random_generator_seed']

    if params['stratify_var'] is not None:
        stratify = y_cat.loc[:, params['stratify_var']]
    else:
        stratify = None

    if stratify is not None:
        # We remove the classes with less 2 elements so we can stratify
        repeated_cols = (y.sum(axis = 0) > 1) & (y.isin([0,1])).all(axis=0) | (~y.isin([0,1])).any(axis=0)
        repeated_index = stratify.duplicated(keep=False)

        stratify = stratify[repeated_index]
        X = X[repeated_index]
        y = y.loc[repeated_index, repeated_cols.values]
        eX = eX[repeated_index]
        ey = ey.loc[repeated_index, repeated_cols.values]
        y_cat = y_cat[repeated_index]
        output = output[repeated_index]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = params['validation_sample_size'], random_state=random_state, stratify = stratify)
    X_train_error, X_valid_error, y_train_error, y_valid_error = train_test_split(eX, ey, test_size = params['validation_sample_size'], random_state=random_state, stratify = stratify)
    valid_output = train_test_split(output, test_size = params['validation_sample_size'], random_state=random_state, stratify = stratify)[1]
    y_cat_train, y_cat_valid = train_test_split(y_cat, test_size = params['validation_sample_size'], random_state=random_state, stratify = stratify)

    X_valid = X_valid.reset_index(drop=True)
    X_valid_error = X_valid_error.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    y_valid_error = y_valid_error.reset_index(drop=True)
    valid_output = valid_output.reset_index(drop=True)

    target_names = y_train.columns.to_list()

    return X_train, X_train_error, y_train, y_train_error, y_cat_train, target_names, X_valid, X_valid_error, y_valid, y_valid_error, y_cat_valid, valid_output


def add_inner_title(ax, title, loc, size=None, color=None, rotation=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    if size is None:
        prop = dict(size=plt.rcParams['legend.fontsize'])
    else:
        prop = size
    if color is not None:
        prop['color'] = color
    if rotation is not None:
       prop['rotation'] = rotation
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


def plot_predictions(predictions, y_test, names, used_photo):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.close('all')

    fig, axs = plt.subplots(2, len(predictions), gridspec_kw={'height_ratios': [2, 1]}, figsize=(10, 6))

    for (ax, name) in zip(axs[0].flatten(), names):

        y_pred = predictions.loc[:, name]
        y_true = y_test.loc[:, name]

        lim = (y_true.max() - y_true.min()) * 0.25
        x_res = (y_true+y_pred)/np.sqrt(2)
        y_res = (y_pred-y_true)/np.sqrt(2)

        divider = make_axes_locatable(ax)
        # below height and pad are in inches
        ax_ress = divider.append_axes("bottom", 0.75, pad=0.5)
        #ax.xaxis.set_tick_params(labelbottom=False)

        locx, locy, hh, hh_filt, hh_d, hh_d_filt, xcenters, ycenters, lwdx, lwdy, lwd_d = plot_density(y_true, y_pred, z = None, xyrange = [xlim, ylim], thresh = 5, bins = [100, 100])

        contour_levels = np.linspace(np.nanmin(np.log10(hh_filt)), np.nanmax(np.log10(hh_filt))*0.9, 5)
        cs = ax.contour(xcenters, ycenters, np.log10(hh_filt.T), colors = 'w', linestyles ='-', linewidths = 0.85, levels = contour_levels, zorder = 3)

        hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap='inferno_r', bins = 'log', extent= [y_true.min(), y_true.max(), y_true.min(), y_true.max()])
        hb_res = ax_ress.hexbin(x_res, y_res, gridsize=50, cmap='inferno_r', bins = 'log', extent= [x_res.min(), x_res.max(), y_res.min(), y_res.max()])

        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '-')
        ax.set_xlim(y_true.min(), y_true.max())
        ax.set_ylim(y_true.min(), y_true.max())

        ax_ress.axhline(y=0)
        ax_ress.set_xlim(x_res.min(), x_res.max())
        ax_ress.set_ylim(y_res.min(), y_res.max())

        ax_ress.set_xlabel('(%s$_{true}$+%s$_{pred})/\sqrt{2}$'%(name, name))
        ax_ress.set_ylabel('(%s$_{pred}$-%s$_{true})/\sqrt{2}$'%(name, name))
        ax.set_ylabel('%s$_{pred}$'%name)
        ax.set_xlabel('%s$_{true}$'%name)

    for (ax, name) in zip(axs[1].flatten(), names):

        y_pred = predictions.loc[:, name]
        y_true = y_test.loc[:, name]

        lim = (y_true.max() - y_true.min()) * 0.15
        y_res = (y_pred-y_true)/np.sqrt(2)

        #ax.hist(y_true-y_pred, 50, range = [-lim, lim])
        ax.hist(y_res, 50, range = [-lim, lim])
        ax.axvline(x=0, color = 'r')
        ax.set_xlabel('%s$_{true}$ - %s$_{pred}$'%(name, name))
        #add_inner_title(ax, '$\sigma=%.2f$'%(np.std(y_true-y_pred)), 1, size=None, color=None, rotation=None)
        add_inner_title(ax, '$\sigma=%.2f$'%(np.std(y_res)), 2, size=None, color=None, rotation=None)

    plt.tight_layout()
    plt.savefig('%s_BNN.png'%used_photo)


def plot_density(xdata, ydata, zdata = None, xyrange = None, thresh = 10, bins = [100, 100], kernel_density = 2):
    import scipy
    from scipy import stats
    from astropy.convolution import convolve
    from astropy.convolution import Gaussian2DKernel

    #histogram definition
    if xyrange is None:
        xyrange = [[xdata.min(), xdata.max()], [ydata.min(), xdata.max()]] # data range

    # histogram the data
    hh, locx, locy = scipy.histogram2d(xdata, ydata, range=xyrange, bins=bins)

    kernel = Gaussian2DKernel(kernel_density)
    hh_filt = convolve(hh, kernel)

    xcenters = (locx[:-1] + locx[1:]) / 2
    ycenters = (locy[:-1] + locy[1:]) / 2

    posx = np.digitize(xdata, locx)
    posy = np.digitize(ydata, locy)

    #select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are

    xdata_ld = xdata[ind][hhsub < thresh] # low density points
    ydata_ld = ydata[ind][hhsub < thresh]
    #hh_filt[hh < thresh] = np.nan # fill the areas with low density by NaNs

    if zdata is not None:
       weighted = stats.binned_statistic_2d(xdata, ydata, zdata, range=xyrange, bins=bins, statistic='median')[0]
       weighted_filt = convolve(weighted, kernel)
       zdata_ld = zdata[ind][hhsub < thresh]
       weighted_filt[hh < thresh] = np.nan
       weighted[hh < thresh] = np.nan

    return locx, locy, hh, hh_filt, weighted, weighted_filt, xcenters, ycenters, xdata_ld, ydata_ld, zdata_ld


def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.

    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.

    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''

    from scipy.sparse import coo_matrix

    tpr_list = [0]+[None]*len(y_proba)
    fpr_list = [0]+[None]*len(y_proba)
    for i in range(len(y_proba)):

        TN, FP, FN, TP = coo_matrix( (np.ones(y_real.shape[0], dtype=np.int64), (y_real, y_proba >= y_proba[i])), shape=(2, 2)).toarray().ravel()

        tpr_list[i+1] = TP/(TP + FN) # sensitivity - true positive rate
        fpr_list[i+1] = 1 - TN/(TN+FP) # 1-specificity - false positive rate

    return tpr_list, fpr_list


def plot_predictions_nominal(predictions_nominal, losses_nominal, pdfs, params, losses_variance = None, xlims = [[0,1], [3000, 8000], [0, 5], [-3, 1]]):

    # Lets plot the variance divided by the error

    from astropy.stats import sigma_clip, sigma_clipped_stats, mad_std
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, RocCurveDisplay, roc_auc_score
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import seaborn as sns
    import matplotlib.ticker as ticker

    stats='pc50'
    stats_var ='std'

    output_directory = params['variables_n_preproc']['output_path']+params['experiment_name']

    y_obs = params['variables_n_preproc']['targets_names']
    y_obs_err = params['variables_n_preproc']['targets_error_names']

    y_pred_nom = ['%s_%s'%(var, stats) for var in params['model_nominal']['targets_names']]
    y_pred_nom_pdf = ['%s'%var for var in params['model_nominal']['targets_names']]

    y_pred_nom_pdf_regres = [col for col in y_pred_nom_pdf if (('CLASS' not in col) and ('class' not in col))]

    y_pred_nom_pdf_class = [col for col in y_pred_nom_pdf if ( (('CLASS' in col) or ('class' in col)) and (('SUBCLASS' not in col) and ('subclass' not in col)) )]
    y_obs_class = [col for col in y_obs if ( (('CLASS' in col) or ('class' in col)) and (('SUBCLASS' not in col) and ('subclass' not in col)) )]
    y_pred_nom_class = [col for col in y_pred_nom if ( (('CLASS' in col) or ('class' in col)) and (('SUBCLASS' not in col) and ('subclass' not in col)) )]

    y_obs_regres = [col for col in y_obs if (('CLASS' not in col) and ('class' not in col))]
    y_obs_err_regres = [col for col in y_obs_err if (('CLASS' not in col) and ('class' not in col))]
    y_pred_nom_regres = [col for col in y_pred_nom if (('CLASS' not in col) and ('class' not in col))]

    y_pred_nom_pdf_subclass = [col for col in y_pred_nom_pdf if (('SUBCLASS' in col) or ('subclass' in col))]
    y_obs_subclass = [col for col in y_obs if (('SUBCLASS' in col) or ('subclass' in col))]
    y_pred_nom_subclass = [col for col in y_pred_nom if (('SUBCLASS' in col) or ('subclass' in col))]

    if params['experiment']['fit_uncertainty_cv']:
        y_pred_var = ['%s_%s'%(var, stats) for var in params['model_variance']['targets_names']]
        y_pred_var_pdf = ['%s'%var for var in params['model_variance']['targets_names']]
        y_pred_var_regres = [col for col in y_pred_var if (('CLASS' not in col) and ('class' not in col))]
        y_pred_var_class = [col for col in y_pred_var if ( (('CLASS' in col) or ('class' in col)) and (('SUBCLASS' not in col) and ('subclass' not in col)) )]
        y_pred_var_subclass = [col for col in y_pred_var if (('SUBCLASS' in col) or ('subclass' in col))]
    else:
        y_pred_var = ['%s_%s'%(var, stats_var) for var in params['model_nominal']['targets_names']]
        y_pred_var_regres = [col for col in y_pred_var if (('CLASS' not in col) and ('class' not in col))]
        y_pred_var_class = [col for col in y_pred_var if ( (('CLASS' in col) or ('class' in col)) and (('SUBCLASS' not in col) and ('subclass' not in col)) )]
        y_pred_var_subclass = [col for col in y_pred_var if (('SUBCLASS' in col) or ('subclass' in col))]

    mag_list = [col for col in params['variables_n_preproc']['validation_vars'] if (('mag' in col) and ('_err' not in col) and ('e_' not in col) and ('_magerr' not in col))]
    mag_err_list = [col for col in params['variables_n_preproc']['validation_vars'] if (('mag' in col) and (('_err' in col) or ('e_' in col) or ('_magerr' in col)))]
    residual_list = ['%s_residual'%col for col in y_pred_nom]
    residual_abs_list = ['%s_residual_abs'%col for col in y_pred_nom]

    residual_reg_list = ['%s_residual'%col for col in y_pred_nom_regres]
    residual_reg_abs_rel_list = ['%s_residual_rel'%col for col in y_pred_nom_regres]

    predictions_nominal[residual_list] = predictions_nominal.loc[:, y_pred_nom] - predictions_nominal.loc[:, y_obs].values

    predictions_nominal[residual_abs_list] = predictions_nominal.loc[:, residual_list].abs()

    predictions_nominal[residual_reg_abs_rel_list] = predictions_nominal[residual_reg_list].abs()/(1+predictions_nominal[y_obs_regres].values)

    # We fill the Nans
    predictions_nominal = predictions_nominal.fillna(99)

    # We clip the results
    predictions_nominal.loc[:, mag_list] = predictions_nominal.loc[:, mag_list].clip(0, 25)
    predictions_nominal.loc[:, mag_err_list] = predictions_nominal.loc[:, mag_err_list].clip(0, 0.75)

    # We form the lists for the x and y variables.
    xx_list = [mag_list, mag_err_list, mag_list, mag_err_list, mag_list, mag_err_list]
    yy_list = [residual_list, residual_list, y_pred_var, y_pred_var]
    zz_list = [y_pred_var, y_pred_var, y_pred_var, y_pred_var]

    #try:
    classes_combinations = []
    class_list = y_obs_class
    for i in range(len(class_list)):
        for j in range(i+1, len(class_list)):
            classes_combinations.append([class_list[i], class_list[j]])
            classes_combinations.append([class_list[j], class_list[i]])

    mag = 'magab_mag_auto_rSDSSB'
    mag_limit = 21.5
    above_limit = predictions_nominal[mag] < mag_limit

    true_class = np.argmax(predictions_nominal.loc[above_limit, y_obs_class].values, axis = 1)

    # Plots the Probability Distributions and the ROC Curves One vs One
    plt.close('all')
    plt.figure(figsize = (14, 5))
    #plt.suptitle(r'%s $\leq$ %.1f. Execution time %.1f'%(mag, mag_limit, params['experiment']['execution_time']))

    bins = [i/20 for i in range(20)] + [1]
    roc_auc_ovo = {}
    for i in range(len(classes_combinations)):
        # Gets the class
        comb = classes_combinations[i]
        c1 = comb[0]
        c2 = comb[1]
        c1_index = class_list.index(c1)
        title = c1 + " vs " +c2

        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame()
        df_aux['class'] = [y_obs_class[ii] for ii in true_class]
        df_aux['prob'] = predictions_nominal.loc[above_limit, y_pred_nom_class].values[:, c1_index]

        # Slices only the subset with both classes
        df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
        df_aux['class'] = [1 if y == c1 else 0 for y in df_aux['class']]
        df_aux = df_aux.reset_index(drop = True)

        # Plots the probability distribution for the class and the rest
        ax = plt.subplot(2, 6, i+1)
        sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins, stat = "probability")
        if i == 0:
            ax.set_ylabel('Normalized count')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

        ax.set_title(title, fontdict={'fontsize':'8'})
        ax.legend([f"{c1}", f"{c2}"], prop={'size':'9'})
        ax.set_xlabel(f"P(x = {c1})")
        ax.grid(which='both')

        # Calculates the ROC Coordinates and plots the ROC Curves
        ax_bottom = plt.subplot(2, 6, i+7)
        tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])

        sns.lineplot(x = fpr, y = tpr, ax = ax_bottom, color = 'C0')
        ax_bottom.plot([0, 1], [0, 1], 'r--')
        ax_bottom.set_xlim(-0.01, 1.01)
        ax_bottom.set_ylim(-0.01, 1.01)
        if i == 0:
            ax_bottom.set_ylabel("True Positive Rate")
        else:
            ax_bottom.set_yticklabels([])

        ax_bottom.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_bottom.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_bottom.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax_bottom.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

        ax_bottom.set_xlabel("False Positive Rate")
        ax_bottom.grid(which='both')

        # Calculates the ROC AUC OvO
        roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])
        add_inner_title(ax_bottom, r'ROC AUC = %.3f'%roc_auc_ovo[title], 4, size={'size':'9'}, color=None, rotation=None)

    plt.subplots_adjust(left = 0.1, right=0.9, wspace = 0.1, hspace = 0.35, bottom=0.1)
    plt.savefig('%s/%s_ROC_curves.png'%(output_directory, params['experiment_name']), bbox_inches='tight')
    #except:
        #pass

    try:
        # Lets make histograms with the residuals versus the magnitudes
        bins = np.arange(17, 26, 1.0)
        binscenters = (bins[:-1] + bins[1:]) / 2
        mag = 'magab_mag_auto_rSDSSB'
        predictions_nominal['mag_bin'] = pd.cut(x = predictions_nominal[mag], bins = bins)

        for ii, (group_name, group) in enumerate(predictions_nominal.groupby(['mag_bin'])):
            if len(group) > 50:

                # Lets make histograms with the residuals versus the magnitudes
                true_class = np.argmax(group.loc[:, y_obs_class].values, axis = 1)
                class_list = [[clase for clase in y_obs_class][i] for i in list(set(true_class))]

                classes_combinations = []
                class_list = y_obs_class
                for i in range(len(class_list)):
                    for j in range(i+1, len(class_list)):
                        classes_combinations.append([class_list[i], class_list[j]])
                        classes_combinations.append([class_list[j], class_list[i]])

                # Plots the Probability Distributions and the ROC Curves One vs One
                plt.close('all')
                fig = plt.figure(figsize = (14, 5))
                #plt.suptitle(r'%s < %s $\leq$ %s. Execution time %.1f sec.'%(group_name.left, mag, group_name.right, params['experiment']['execution_time']))
                bins = [i/20 for i in range(20)] + [1]
                roc_auc_ovo = {}
                for i in range(len(classes_combinations)):
                    # Gets the class
                    comb = classes_combinations[i]
                    c1 = comb[0]
                    c2 = comb[1]
                    c1_index = class_list.index(c1)
                    title = c1 + " vs " +c2

                    # Prepares an auxiliar dataframe to help with the plots
                    df_aux = pd.DataFrame()
                    df_aux['class'] = [y_obs_class[ii] for ii in true_class]
                    df_aux['prob'] = group.loc[:, y_pred_nom_class].values[:, c1_index]

                    # Slices only the subset with both classes
                    df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
                    df_aux['class'] = [1 if y == c1 else 0 for y in df_aux['class']]
                    df_aux = df_aux.reset_index(drop = True)

                    # Plots the probability distribution for the class and the rest
                    ax = plt.subplot(2, 6, i+1)
                    sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins, stat = "probability")
                    if i == 0:
                        ax.set_ylabel('Normalized count')
                    else:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])
                    ax.set_ylim(0, 1.0)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
                    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

                    ax.set_title(title, fontdict={'fontsize':'8'})
                    ax.legend([f"{c1}", f"{c2}"], prop={'size':'9'})
                    ax.set_xlabel(f"P(x = {c1})")
                    ax.grid(which='both')

                    # Calculates the ROC Coordinates and plots the ROC Curves
                    ax_bottom = plt.subplot(2, 6, i+7)
                    tpr, fpr = get_all_roc_coordinates(df_aux['class'], df_aux['prob'])

                    sns.lineplot(x = fpr, y = tpr, ax = ax_bottom, color = 'C0')
                    ax_bottom.plot([0, 1], [0, 1], 'r--')
                    ax_bottom.set_xlim(-0.01, 1.01)
                    ax_bottom.set_ylim(-0.01, 1.01)
                    if i == 0:
                        ax_bottom.set_ylabel("True Positive Rate")
                    else:
                        ax_bottom.set_yticklabels([])

                    ax_bottom.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax_bottom.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                    ax_bottom.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
                    ax_bottom.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

                    ax_bottom.set_xlabel("False Positive Rate")
                    ax_bottom.grid(which='both')

                    # Calculates the ROC AUC OvO
                    roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])
                    add_inner_title(ax_bottom, r'ROC AUC = %.3f'%roc_auc_ovo[title], 4, size={'size':'9'}, color=None, rotation=None)
                    #add_inner_title(ax1, r'%s < %s $\leq$ %s'%(group_name.left, mag, group_name.right), 2, size={'size':'9'}, color=None, rotation=None)

                plt.subplots_adjust(left = 0.1, right=0.9, wspace = 0.1, hspace = 0.35, bottom=0.1)
                plt.savefig('%s/%s_ROC_curves_%s_%s.png'%(output_directory, params['experiment_name'], mag, str(group_name).strip('(]').replace(', ', '_')), bbox_inches='tight')
    except:
        pass

    try:
        plt.close('all')
        loss, val_loss = losses_nominal['loss'], losses_nominal['val_loss']
        plt.plot(loss, '--')
        plt.plot(val_loss, '-')
        plt.legend(['loss', 'val loss'])
        plt.ylabel('loss')
        plt.xlabel('Epochs')
        plt.savefig('%s/%s_loss_nominal.png'%(output_directory, params['experiment_name']), bbox_inches='tight')
    except:
        pass

    try:
        plt.close('all')
        loss, val_loss = losses_variance['loss'], losses_variance['val_loss']
        plt.plot(loss, '--')
        plt.plot(val_loss, '-')
        plt.legend(['loss', 'val loss'])
        plt.ylabel('loss')
        plt.xlabel('Epochs')
        plt.savefig('%s/%s_loss_variance.png'%(output_directory, params['experiment_name']), bbox_inches='tight')
    except:
        pass


    for xx_names, yy_names, zz_names, in zip(xx_list, yy_list, zz_list):
        for yy_name, zz_name in zip(yy_names, zz_names):

            #Quantiles:
            quantiles_x = predictions_nominal[xx_names].quantile([0.1, 1])
            quantiles_y = predictions_nominal[yy_name].quantile([0.01, 0.99])
            quantiles_z = predictions_nominal[zz_name].quantile([0.01, 0.99])

            xlim = [np.nanmin(quantiles_x), np.nanmax(quantiles_x)+1e-3]
            zlim = [np.nanmin(quantiles_z), np.nanmax(quantiles_z)]

            if 'CLASS' in yy_name:
                ylim = [-1.02, 1.02]
            else:
                ylim = [np.nanmin(quantiles_y), np.nanmax(quantiles_y)]

            plt.close('all')
            fig, axs = plt.subplots(5, 4, dpi=200, figsize = [10, 10], sharey = True, sharex = True)
            for ii, (ax, xx_name) in enumerate(zip(axs.flatten(), xx_names)):

                locx, locy, hh, hh_filt, hh_d, hh_d_filt, xcenters, ycenters, lwdx, lwdy, lwd_d = plot_density(predictions_nominal[xx_name], predictions_nominal[yy_name], zdata = predictions_nominal[zz_name], xyrange = [xlim, ylim], thresh = 1, bins = [100, 100])

                X, Y = np.meshgrid(locx, locy)
                ax.pcolormesh(X, Y, hh_d.T)

                contour_levels = np.linspace(np.nanmin(np.log10(hh_filt)), np.nanmax(np.log10(hh_filt))*0.9, 5)
                cs = ax.contour(xcenters, ycenters, np.log10(hh_filt.T), colors = 'w', linestyles ='-', linewidths = 0.85, levels = contour_levels, zorder = 3)

                cb = ax.scatter(lwdx, lwdy, c = lwd_d, s = 1, alpha = 0.75, vmin = zlim[0], vmax = zlim[1])
                ax.set_xlabel(xx_name)
                ax.axhline(0, color = 'r', linewidth = 0.8)
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)
                ax.grid()
            plt.subplots_adjust(left = 0.1, right=0.9, wspace = 0.25, hspace = 0.5, bottom=0.15)

            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.ylabel(yy_name.replace('pred_nom_',''))

            # Colorbar
            cbar = fig.colorbar(cb, ax=axs.ravel().tolist(), aspect = 40, pad = 0.02)
            cbar.set_label(zz_name.replace('pred_nom_',''), labelpad = -0.05)

            if any('mag_err' in xx_name for xx_name in xx_names):
                plt.savefig('%s/%s_%s_vs_mag_err.png'%(output_directory, params['experiment_name'], yy_name.replace('pred_nom_','')), bbox_inches='tight')
            else:
                plt.savefig('%s/%s_%s_vs_mag.png'%(output_directory, params['experiment_name'], yy_name.replace('pred_nom_','')), bbox_inches='tight')


    try:
        # Lets make histograms with the residuals versus the magnitudes
        predicted_class = np.argmax(predictions_nominal.loc[:, y_pred_nom_class].values, axis = 1)
        true_class = np.argmax(predictions_nominal.loc[:, y_obs_class].values, axis = 1)
        predictions_nominal['missed_class'] = predicted_class != true_class

        bins = np.arange(17, 26, 1.0)
        binscenters = (bins[:-1] + bins[1:]) / 2

        for mag in mag_list:
            predictions_nominal['mag_bin'] = pd.cut(x = predictions_nominal[mag], bins = bins)

            class_fails = predictions_nominal.groupby(['mag_bin'])['missed_class'].sum()
            counts = predictions_nominal.groupby(['mag_bin'])['missed_class'].count()

            failure_ratio = 100*class_fails/counts
            relative_failure_ratio = failure_ratio/failure_ratio.sum()
            cumulative = 100*class_fails.cumsum()/counts.sum()

            plt.close('all')
            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 4))
            cm = confusion_matrix(true_class, predicted_class, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[clase.replace('CLASS_', '') for clase in y_obs_class])
            disp.plot(ax = ax1)
            ax2.bar(binscenters, failure_ratio, width = 1, edgecolor = 'k')
            ax2.step(bins[1::], cumulative, linestyle = '--', color = 'C1', label = 'Cumulative')
            ax2.set_xlabel(r'%s [mag]'%mag)
            ax2.set_ylabel(r'Classification relative error [%]')
            ax2.legend()
            ax2.grid()
            ax2.set_ylim([0, 25])
            plt.subplots_adjust(left = 0.1, right=0.9, wspace = 0.25, hspace = 0.5, bottom=0.15)
            plt.savefig('%s/%s_class_ANN_%s.png'%(output_directory, params['experiment_name'], mag), bbox_inches='tight')
    except:
        pass


    try:
        # Lets make histograms with the residuals versus the magnitudes
        ylim = 25

        predicted_class = np.argmax(predictions_nominal.loc[:, y_pred_nom_class].values, axis = 1)
        true_class = np.argmax(predictions_nominal.loc[:, y_obs_class].values, axis = 1)
        predictions_nominal['missed_class'] = predicted_class != true_class

        bins = np.arange(17, 26, 1.0)
        binscenters = (bins[:-1] + bins[1:]) / 2

        mag = 'magab_mag_auto_rSDSSB'
        predictions_nominal['mag_bin'] = pd.cut(x = predictions_nominal[mag], bins = bins)

        class_fails_all = predictions_nominal.groupby(['mag_bin'])['missed_class'].sum()
        counts_all = predictions_nominal.groupby(['mag_bin'])['missed_class'].count()

        failure_ratio_all = 100*class_fails_all/counts_all
        cumulative_all = 100*class_fails_all.cumsum()/counts_all.sum()

        plt.close('all')
        for group_name, group in predictions_nominal.groupby(['mag_bin']):
            if len(group) > 2:
                # Lets make histograms with the residuals versus the magnitudes
                predicted_class = np.argmax(group.loc[:, y_pred_nom_class].values, axis = 1)
                true_class = np.argmax(group.loc[:, y_obs_class].values, axis = 1)
                group['missed_class'] = predicted_class != true_class

                class_fails = group.groupby(['mag_bin'])['missed_class'].sum()
                counts = group.groupby(['mag_bin'])['missed_class'].count()

                failure_ratio = 100*class_fails/counts
                relative_failure_ratio = failure_ratio/failure_ratio.sum()
                cumulative = 100*class_fails.cumsum()/counts.sum()

                labels = [[clase.replace('CLASS_', '') for clase in y_obs_class][i] for i in list(set(true_class))]

                plt.close('all')
                fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 4))
                cm = confusion_matrix(true_class, predicted_class, normalize='true')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels)
                disp.plot(ax = ax1)
                add_inner_title(ax1, r'%s < %s $\leq$ %s'%(group_name.left, mag, group_name.right), 2, size={'size':'9'}, color=None, rotation=None)

                add_inner_title(ax1, '%s objects'%len(group), 3, size={'size':'9'}, color=None, rotation=None)

                err = ax2.bar(binscenters, failure_ratio_all, width = 1, edgecolor = 'k', label = 'Mag bin error [%]')
                cum = ax2.step(bins[1::], cumulative_all, linestyle = '--', color = 'C3', linewidth = 2, label = 'Total cumulative error [%]')

                ax3 = ax2.twinx()
                no = ax3.step(bins[1::], counts_all, linestyle = ':', linewidth = 2, color = 'k', label = 'N. objects')
                ax3.set_ylabel('N. objects')

                ax2.bar(binscenters, failure_ratio, width = 1, edgecolor = 'r')

                ax2.set_xlabel(r'%s [mag]'%mag)
                ax2.set_ylabel(r'Classification error [%]')

                ax2.legend(loc = 2)

                ax2.grid()
                ax2.set_ylim([0, ylim])

                #add_inner_title(ax2, '%s objects'%len(group), 2, size=None, color=None, rotation=None)

                plt.subplots_adjust(left = 0.1, right=0.9, wspace = 0.25, hspace = 0.5, bottom=0.15)

            plt.savefig('%s/%s_class_ANN_%s_%s.png'%(output_directory, params['experiment_name'], mag, str(group_name).strip('(]').replace(', ', '_')), bbox_inches='tight')
    except:
        pass


    # The collapsed version
    try:
        # Lets make histograms with the residuals versus the magnitudes
        bins = np.arange(17, 24, 1.0)
        binscenters = (bins[:-1] + bins[1:]) / 2
        mag = 'magab_mag_auto_rSDSSB'
        predictions_nominal['mag_bin'] = pd.cut(x = predictions_nominal[mag], bins = bins)

        plt.close('all')
        fig, axes = plt.subplots(2, 3, figsize = (10, 6), sharex = True, sharey = True)
        plt.suptitle(r'Execution time %.1f'%(params['experiment']['execution_time']))

        for ii, ((group_name, group), ax) in enumerate(zip(predictions_nominal.groupby(['mag_bin']), axes.flatten())):
            if len(group) > 0:
                # Lets make histograms with the residuals versus the magnitudes
                predicted_class = np.argmax(group.loc[:, y_pred_nom_class].values, axis = 1)
                true_class = np.argmax(group.loc[:, y_obs_class].values, axis = 1)
                group['missed_class'] = predicted_class != true_class

                labels = [[clase.replace('CLASS_', '') for clase in y_obs_class][i] for i in list(set(true_class))]
                cm = confusion_matrix(true_class, predicted_class, normalize='true')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = labels)
                disp.plot(ax = ax, colorbar = False, im_kw = {'vmin':0, 'vmax':1})
                add_inner_title(ax, r'%s < %s $\leq$ %s'%(group_name.left, mag, group_name.right), 2, size={'size':'9'}, color=None, rotation=None)
                #add_inner_title(ax, '%s objects'%len(group), 3, size={'size':'9'}, color=None, rotation=None)

                if (ii)%3:
                    ax.set_ylabel('')
                if ii <= 7:
                    ax.set_xlabel('')

                plt.subplots_adjust(left = 0.1, right=0.9, wspace = 0.1, hspace = 0.1, bottom=0.1)
        plt.savefig('%s/%s_class_ANN_permagbin.png'%(output_directory, params['experiment_name']), bbox_inches='tight')
    except:
        pass


    try:
        bins = np.arange(17, 26, 1.0)
        binscenters = (bins[:-1] + bins[1:]) / 2

        for residual_rel in residual_reg_abs_rel_list:
            plt.close('all')
            fig, axs = plt.subplots(5, 4, dpi=200, figsize = [10, 10], sharey = True, sharex = True)
            for ii, (ax, mag) in enumerate(zip(axs.flatten(), mag_list)):
                predictions_nominal['mag_bin'] = pd.cut(x = predictions_nominal[mag], bins = bins)
                agregated_median = predictions_nominal.groupby(['mag_bin'])[residual_rel].median()
                agregated_mean = predictions_nominal.groupby(['mag_bin'])[residual_rel].mean()
                agregated_std = predictions_nominal.groupby(['mag_bin'])[residual_rel].std()
                cumulative = (agregated_median.cumsum()/agregated_median.sum())

                ax.bar(binscenters, agregated_median, width = 1, edgecolor = 'k')
                ax.axvline(x = 25, color = 'r')

                ax2 = ax.twinx()
                if (ii+1)%4:
                    ax2.set_yticklabels([])

                ax2.step(bins[1::], cumulative, linestyle = '--', color = 'C1', label = 'Cumulative')
                ax2.tick_params(axis='y', labelcolor='C1')

                ax.set_xlabel(r'%s [mag]'%mag)
                ax.set_xlim([binscenters[0], binscenters[-1]])
                ax.set_ylim([0, 0.25])
                ax2.set_ylim([0,1])
                ax.grid()

            plt.subplots_adjust(left = 0.1, right=0.9, wspace = 0.25, hspace = 0.5, bottom=0.15)

            ax_o = fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            ax_o.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

            ax_o.set_ylabel(r'$\Delta(z)/(1+z)$')

            ax3 = ax_o.twinx()
            ax3.spines["left"].set_visible(False)
            ax3.spines["bottom"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            ax3.spines["top"].set_visible(False)
            ax3.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            ax3.set_ylabel('cumulative', color='C1')  # we already handled the x-label with ax1

            plt.savefig('%s/%s_regression_relerr_%s.png'%(output_directory, params['experiment_name'], residual_rel), bbox_inches='tight')
    except:
        pass

    for obs, pred_nom, pred_var in zip(y_obs, y_pred_nom_regres, y_pred_var_regres):
        plt.close('all')

        variance = (predictions_nominal[obs] - predictions_nominal[pred_nom].values) / predictions_nominal[pred_var].values

        clipped_variance = np.ma.filled(sigma_clip(variance, sigma=6, maxiters=None))
        mu = np.nanmean(clipped_variance)
        m = np.nanmedian(clipped_variance)
        std = mad_std(clipped_variance)

        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
        ax.hist(variance, 50, density=True, range = [-6, 6])

        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 200)

        std = mad_std(variance)
        print(m, std)
        p = norm.pdf(x, m, std)
        ax.plot(x, p, 'k', linewidth=1.5)
        add_inner_title(ax, '$\mu=%.2f, \sigma=%.2f$'%(m, std), 2, size=None, color=None, rotation=None)

        plt.savefig('%s/%s_%s_distro_errors.png'%(output_directory, params['experiment_name'], obs), bbox_inches='tight')


    for ii, (pdf, idx) in enumerate(zip(pdfs, params['variables_n_preproc']['saved_pdfs_indexes'])):
        try:
            plt.close('all')
            fig, axs = plt.subplots(1, len(y_obs_regres) + 1, dpi=300, figsize = [14, 3])
            pred_data = predictions_nominal.iloc[idx, :]

            for (ax, obs, pred_nom, pred_var, pred_nom_pdf) in zip(axs[0:-1], y_obs_regres, y_pred_nom_regres, y_pred_var_regres, y_pred_nom_pdf, ):

                ax.hist(pdf[pred_nom_pdf], 50, histtype = 'step')
                ax.axvline(pred_data[pred_nom], color = 'C0', linestyle = '-', linewidth = 0.8)
                ax.axvspan(pred_data[pred_nom] - pred_data[pred_var], pred_data[pred_nom] + pred_data[pred_var], color = 'C0', alpha = 0.2)

                ax.axvline(pred_data[obs], color = 'r', linestyle = '--', linewidth = 0.8, zorder = 3)
                ax.set_xlabel(obs)
                ax.set_yticklabels([])

            axs[-1].hist(pdf.loc[:, y_pred_nom_pdf_class], 50, range = [-0.1, 1.1], histtype = 'step', label = y_obs_class)
            axs[-1].legend()
            plt.savefig('%s/%s_PDF_%i.png'%(output_directory, params['experiment_name'], idx), bbox_inches='tight')
        except:
            pass

    # The photo Z per mag bin
    try:
        # Lets make histograms with the residuals versus the magnitudes
        bins = np.arange(13, 26, 1.0)
        binscenters = (bins[:-1] + bins[1:]) / 2
        mag = 'magab_mag_auto_rSDSSB'
        predictions_nominal['mag_bin'] = pd.cut(x = predictions_nominal[mag], bins = bins)

        for obs_regres, obs_err_regres, pred_nom_regres, pred_var_regres in zip(y_obs_regres, y_obs_err_regres, y_pred_nom_regres, y_pred_var_regres):
            xlims = predictions_nominal[obs_regres].quantile([0.01, 0.99]).values
            ylims = xlims
            zlims = predictions_nominal[pred_var_regres].quantile([0.01, 0.99]).values
            limsy = [-0.3, 0.3]
            thresh = 2

            plt.close('all')
            fig, axes = plt.subplots(3, 4, figsize = (12, 10), sharex = True, sharey = True)
            for ii, ((group_name, group), ax) in enumerate(zip(predictions_nominal.groupby(['mag_bin']), axes.flatten())):
                if len(group) > 0:
                    # Lets make histograms with the residuals versus the magnitudes
                    locx, locy, hh, hh_filt, hh_d, hh_d_filt, xcenters, ycenters, lwdx, lwdy, lwd_d = plot_density(group[obs_regres], group[pred_nom_regres], zdata = group[pred_var_regres], xyrange = [xlims, ylims], thresh = thresh, bins = [100, 100])

                    X, Y = np.meshgrid(locx, locy)
                    ax.pcolormesh(X, Y, hh_d.T)

                    contour_levels = np.linspace(np.log10( max( [np.nanmin(hh_filt), thresh] )), max( [np.nanmax(np.log10(hh_filt))*0.99, thresh + 1]), 10)
                    cs = ax.contour(xcenters, ycenters, np.log10(hh_filt.T), colors = 'w', linestyles ='-', linewidths = 0.85, levels = contour_levels, zorder = 3)

                    cb = ax.scatter(lwdx, lwdy, c = lwd_d, s = 1, alpha = 0.75, vmin = zlims[0], vmax = zlims[1])
                    ax.plot(xlims, xlims, color = 'r', linewidth = 0.8)
                    ax.set_ylim(ylims)
                    ax.set_xlim(xlims)
                    ax.grid()
                    ax.set_xlabel(obs_regres)
                    ax.set_ylabel(pred_nom_regres.replace('_nom', '').replace('_median', ''))

                    add_inner_title(ax, r'%s < %s $\leq$ %s'%(group_name.left, mag, group_name.right), 2, size={'size':'9'}, color=None, rotation=None)
                    add_inner_title(ax, '%s objects'%len(group), 3, size={'size':'9'}, color=None, rotation=None)

                    if (ii)%4:
                        ax.set_ylabel('')
                    if ii <= 7:
                        ax.set_xlabel('')

                    plt.subplots_adjust(left = 0.1, right=0.9, wspace = 0.1, hspace = 0.1, bottom=0.1)
            plt.savefig('%s/%s_photo_%s_permagbin.png'%(output_directory, params['experiment_name'], obs_regres), bbox_inches='tight')
    except:
        pass


    for obs_regres, obs_err_regres, pred_nom_regres, pred_var_regres in zip(y_obs_regres, y_obs_err_regres, y_pred_nom_regres, y_pred_var_regres):

        if 'Z' in obs_regres:
            xlims = predictions_nominal[obs_regres].quantile([0.01, 0.99]).values
            ylims = xlims
        else:
            xlims = predictions_nominal[obs_regres].quantile([0.01, 0.99]).values
            ylims = xlims

        zlims = predictions_nominal[pred_var_regres].quantile([0.01, 0.99]).values
        limsy = [-0.3, 0.3]
        thresh = 5

        locx, locy, hh, hh_filt, hh_d, hh_d_filt, xcenters, ycenters, lwdx, lwdy, lwd_d = plot_density(predictions_nominal[obs_regres], predictions_nominal[pred_nom_regres], zdata = predictions_nominal[pred_var_regres], xyrange = [xlims, ylims], thresh = thresh, bins = [100, 100])

        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw={'height_ratios': [2, 1]}, sharex = True, figsize = (5, 8))
        X, Y = np.meshgrid(locx, locy)
        ax1.pcolormesh(X, Y, hh_d.T)

        contour_levels = np.linspace(np.log10( max( [np.nanmin(hh_filt), thresh] )), max( [np.nanmax(np.log10(hh_filt))*0.99, thresh + 1]), 10)

        cs = ax1.contour(xcenters, ycenters, np.log10(hh_filt.T), colors = 'w', linestyles ='-', linewidths = 0.85, levels = contour_levels, zorder = 3)

        cb = ax1.scatter(lwdx, lwdy, c = lwd_d, s = 1, alpha = 0.75, vmin = zlims[0], vmax = zlims[1])
        ax1.plot(xlims, xlims, color = 'r', linewidth = 0.8)
        ax1.set_ylim(ylims)
        ax1.set_xlim(xlims)
        ax1.grid()
        ax1.set_ylabel(obs_regres)

        # Colorbar
        cbar = fig.colorbar(cb, ax=ax1, aspect = 15, pad = 0.03)
        cbar.set_label(pred_var_regres, labelpad = +0.5)

        if 'Z' in obs_regres:
            residuals_y = (predictions_nominal[pred_nom_regres].values-predictions_nominal[obs_regres].values)/(1+predictions_nominal[obs_regres].values)
            residuals_x = predictions_nominal[obs_regres]
        else:
            residuals_x = (predictions_nominal[obs_regres]+predictions_nominal[pred_nom_regres]) / np.sqrt(2)
            residuals_y = (predictions_nominal[pred_nom_regres]-predictions_nominal[obs_regres]) / np.sqrt(2)

        stats = sigma_clipped_stats(residuals_y, sigma=6.0, maxiters=None)

        locx, locy, hh, hh_filt, hh_d, hh_d_filt, xcenters, ycenters, lwdx, lwdy, lwd_d = plot_density(residuals_x, residuals_y, zdata = predictions_nominal[pred_var_regres], xyrange = [xlims, limsy], thresh = thresh, bins = [100, 100])

        X, Y = np.meshgrid(locx, locy)
        ax2.pcolormesh(X, Y, hh_d.T)
        cs = ax2.contour(xcenters, ycenters, np.log10(hh_filt.T), colors = 'w', linestyles ='-', linewidths = 0.85, levels = contour_levels, zorder = 3)
        cb = ax2.scatter(lwdx, lwdy, c = lwd_d, s = 1, alpha = 0.75, vmin = zlims[0], vmax = zlims[1])
        ax2.set_xlabel(obs_regres)
        ax2.set_xlim(xlims)
        ax2.axhline(y=0, color = 'r', linewidth = 0.8)
        ax2.grid()

        if 'Z' in obs_regres:
            ax2.set_ylabel(r'$\Delta(z)/(1+z)$')
        else:
            ax2.set_ylabel(r'$\Delta$(%s)'%obs_regres)

        divider = make_axes_locatable(ax2)
        # below height and pad are in inches
        ax_histy = divider.append_axes("right", 0.9, pad=0.2, sharey=ax2)
        ax_histy.yaxis.set_tick_params(labelleft=False)
        ax_histy.hist(residuals_y, bins=100, orientation='horizontal', range = limsy)
        ax_histy.grid()
        add_inner_title(ax_histy, '$\mu=%.2f$\n$m=%.2f$\n$\sigma=%.2f$'%(stats), 2, size=None, color=None, rotation=None)

        plt.subplots_adjust(left = 0.1, right=0.9, wspace = 0.15, hspace = 0.15, bottom=0.15)

        plt.savefig('%s/%s_photo_%s.png'%(output_directory, params['experiment_name'], obs_regres), bbox_inches='tight')


        # We rotate the predictions_nominal to check the dispersion
        predictions_nominal['resx_%s'%obs_regres] = (predictions_nominal[obs_regres]+predictions_nominal[pred_nom_regres]) / np.sqrt(2)
        predictions_nominal['resy_%s'%obs_regres] = (predictions_nominal[pred_nom_regres]-predictions_nominal[obs_regres]) / np.sqrt(2)

        predictions_nominal['res_%s_erru'%obs_regres] = np.sqrt(predictions_nominal[obs_err_regres]**2+predictions_nominal[pred_var_regres]**2)
        predictions_nominal['res_%s_errd'%obs_regres] = np.sqrt(predictions_nominal[obs_err_regres]**2+predictions_nominal[pred_var_regres]**2)

        limsx = np.nanpercentile(predictions_nominal['resx_%s'%obs_regres], [0.1, 99.9], axis=0)

        plt.close('all')
        above =  predictions_nominal['resy_%s'%obs_regres] > 0
        yabove = predictions_nominal.loc[above, 'resy_%s'%obs_regres] - predictions_nominal.loc[above, 'res_%s_errd'%obs_regres]
        ybelow = predictions_nominal.loc[~above, 'resy_%s'%obs_regres] + predictions_nominal.loc[~above, 'res_%s_erru'%obs_regres]

        stats_above = sigma_clipped_stats(yabove, sigma=6.0, maxiters=None)
        stats_below = sigma_clipped_stats(ybelow, sigma=6.0, maxiters=None)

        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)
        ax.plot(predictions_nominal.loc[above, 'resx_%s'%obs_regres], yabove, '.', ms = 1, alpha = 0.5, label = 'above')
        ax.plot(predictions_nominal.loc[~above, 'resx_%s'%obs_regres], ybelow, '.', ms = 1, alpha = 0.5, label = 'below')
        ax.axhline(0, color='C3')
        ax.set_xlim(limsx)
        ax.set_ylim(limsy)
        ax.legend()

        add_inner_title(ax, '$\mu_{above}=%.2f, m_{above}=%.2f, \sigma_{above}=%.2f$\n$\mu_{below}=%.2f, m_{below}=%.2f, \sigma_{below}=%.2f$'%(stats_above[0], stats_above[1], stats_above[2], stats_below[0], stats_below[1], stats_below[2]), 2, size=None, color=None, rotation=None)

        plt.savefig('%s/%s_photo_%s_errors.png'%(output_directory, params['experiment_name'], obs_regres))

