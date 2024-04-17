"""
Libraries module for predict_simple.py
"""

import sys, os
import numpy as np
import pandas as pd
import polars as pl

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import uncertainties.unumpy as unp

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
        del [gmm]

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


def get_data_test(params, dtypes = None, external_cat = None):

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

    X_vars, y_vars, y_vars_err, prediction_vars = params['X_vars'], params['y_vars'], params['y_vars_err'], params['prediction_vars']

    # Read data
    #predict_data = pd.read_csv(params['test_catalog'])
    predict_data = pl.read_csv(params['test_catalog'], dtypes = dtypes).to_pandas()

    print('We have read %i lines.'%len(predict_data))

    if external_cat is not None:
        predict_data = pd.merge(predict_data.rename(columns={'pmra':'pmra_g', 'pmde':'pmde_g', 'e_pmra':'e_pmra_g', 'e_pmde':'e_pmde_g'}), external_cat.reset_index(), how='left', left_on=['tile_id', 'number'], right_on = ['tile_id', 'number'])

    photo, photo_err = get_photometry(predict_data, params['used_photo'])

    #first abs values for PMs
    pmra_g = unp.uarray(predict_data.pmra_g, predict_data.e_pmra_g)
    pmdec_g = unp.uarray(predict_data.pmde_g, predict_data.e_pmde_g)
    pm_g = unp.sqrt(pmra_g**2 + pmdec_g**2)
    predict_data['pm_g_error'] = unp.std_devs(pm_g)
    predict_data['pm_g'] = unp.nominal_values(pm_g)
    X_vars = X_vars + ['pm_g_error', 'pm_g']
    X_vars.remove('e_pmra_g')
    X_vars.remove('e_pmde_g')
    X_vars.remove('pmra_g')
    X_vars.remove('pmde_g')
    del [pmra_g, pmdec_g, pm_g]

    pmra_cw  = unp.uarray(predict_data.pmra_cw*1000, np.abs(predict_data.e_pmra_cw)*1000)
    pmdec_cw = unp.uarray(predict_data.pmde_cw*1000, np.abs(predict_data.e_pmde_cw)*1000)
    pm_cw = unp.sqrt(pmra_cw**2 + pmdec_cw**2)
    predict_data['e_pm_cw'] = unp.std_devs(pm_cw)
    predict_data['pm_cw'] = unp.nominal_values(pm_cw)
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
        c = SkyCoord(ra=predict_data.alpha_j2000.values*u.degree, dec=predict_data.delta_j2000.values*u.degree, frame='fk5')
        predict_data['l'] = c.galactic.l.value
        predict_data['b'] = c.galactic.b.value
        X_vars = X_vars + ['b']

    X_vars = X_vars+photo

    if params['use_photo_error']:
        X_vars = X_vars + photo_err

    if (params['photo_log']) & ('flux' in params['used_photo']):
        # We transform the fluxes to logarithm
        predict_data.loc[:, photo_err] = (predict_data.loc[:, photo_err].values)/((predict_data.loc[:, photo].values)*np.log(10))
        predict_data.loc[:, photo] = np.log10(predict_data.loc[:, photo] - params['fluxes_zpt'])
        try:
            predict_data.loc[:, [ 'e_fg','e_fbp','e_frp']] = (predict_data.loc[:, [ 'e_fg','e_fbp','e_frp']].values)/((predict_data.loc[:, [ 'fg','fbp','frp']].values)*np.log(10))
            predict_data.loc[:, [ 'fg','fbp','frp']] = np.log10(predict_data.loc[:, [ 'fg','fbp','frp']] - params['fluxes_zpt'])
        except:
            pass

    # We are going to add new colors that help with QSO detection
    import itertools
    photo_colors = [mag for mag in predict_data.columns if 'magab_mag_aper_4' in mag]+['w1mpropm_cw', 'w2mpropm_cw']

    # We transform the missing magnitudes to nans
    predict_data.loc[:, photo_colors] = predict_data.loc[:, photo_colors].apply(lambda x: np.where(x == 99.0, np.nan, x)).values

    color_combs = list(itertools.combinations(photo_colors, 2))
    for color_comb in color_combs:
        predict_data['%s-%s'%(color_comb[0], color_comb[1])] = predict_data.loc[:, '%s'%color_comb[0]].values - predict_data.loc[:, '%s'%color_comb[1]].values
        X_vars += ['%s-%s'%(color_comb[0], color_comb[1])]

    # We select the data
    predict_X = predict_data.loc[:, X_vars]
    predict_eX, eX_vars = get_errors(predict_data, used_cols = X_vars)
    predict_output = predict_data.loc[:, prediction_vars]

    # We make sure that -999 are treated as nans, and that there are not nans in the y
    predict_X = predict_X.apply(lambda x: np.where(x < -999, np.nan, x))

    # We apply the same normalization
    non_zero_dynamic_range = np.array(params['norm_dynamic_range']) > 0
    vars_non_zero_dynamic_range = [i for (i, v) in zip(X_vars, non_zero_dynamic_range) if v]
    e_vars_non_zero_dynamic_range = [i for (i, v) in zip(eX_vars, non_zero_dynamic_range) if v]

    predict_X.loc[:, vars_non_zero_dynamic_range] = (predict_X.loc[:, vars_non_zero_dynamic_range] - np.array(params['norm_minimum'])[non_zero_dynamic_range]) / np.array(params['norm_dynamic_range'])[non_zero_dynamic_range]

    predict_eX.loc[:, e_vars_non_zero_dynamic_range] = predict_eX.loc[:, e_vars_non_zero_dynamic_range] / np.array(params['norm_dynamic_range'])[non_zero_dynamic_range]

    predict_X = predict_X.clip(lower = 0, upper = 1)
    predict_eX = predict_eX.clip(lower = 0, upper = 1)

    # We convert the nan to a figure just outside the std of the distribution.
    predict_X = predict_X.fillna(params['fill_na'])
    predict_eX = predict_eX.fillna(0).clip(lower=0)

    return predict_X, predict_eX, predict_output


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

