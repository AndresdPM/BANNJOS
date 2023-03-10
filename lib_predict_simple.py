"""
Libraries module for predict_simple.py
"""

import sys, os
import numpy as np
import pandas as pd

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


def get_statistics(preds, targets_names, index = None):

    from astropy.stats import sigma_clip, sigma_clipped_stats, mad_std
    from sklearn.mixture import GaussianMixture

    clipped_pred = np.ma.filled(sigma_clip(preds, axis = 0, sigma=6, maxiters=10))

    mc_median = np.nanpercentile(clipped_pred, 50.0, axis=0)
    std = mad_std(clipped_pred, axis = 0)
    errd = mc_median - np.nanpercentile(clipped_pred, 15.865, axis=0)
    erru = np.nanpercentile(clipped_pred, 84.135, axis=0) - mc_median
    errdd = mc_median - np.nanpercentile(clipped_pred, 2.275, axis=0)
    erruu = np.nanpercentile(clipped_pred, 97.725, axis=0) - mc_median

    mat = np.array([[(np.sqrt(3)+3)/6, -np.sqrt((2-np.sqrt(3))/6), -1/np.sqrt(3)],
    [-np.sqrt((2-np.sqrt(3))/6), (np.sqrt(3)+3)/6, -1/np.sqrt(3)],
    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])

    rot_preds = np.dot(preds, mat.T)

    gmm_n_comp = 3
    gm = GaussianMixture(n_components=gmm_n_comp)

    rot_covariances = np.ones((preds.shape[1], (preds.shape[2]-1)**2 * gmm_n_comp))*99
    rot_means = np.ones((preds.shape[1], (preds.shape[2]-1) * gmm_n_comp))*99
    rot_weights = np.ones((preds.shape[1], gmm_n_comp))*99

    for ii, rot_pred in enumerate(rot_preds.swapaxes(0,1)):
        cli_progress_test(ii+1, preds.shape[1])
        gm.fit(rot_pred[:,0:2])
        rot_covariances[ii,:] = gm.covariances_.flatten()
        rot_means[ii,:] = gm.means_.flatten()
        rot_weights[ii,:] = gm.weights_.flatten()

    results = pd.DataFrame(columns=[target+'_pc50' for target in targets_names]+[target+'_pc16' for target in targets_names]+[target+'_pc84' for target in targets_names]+[target+'_pc02' for target in targets_names]+[target+'_pc98' for target in targets_names]+[target+'_error' for target in targets_names]+['comp%i_cov_%i%i'%(comp, ii, jj) for comp in [1,2,3] for ii in [1,2] for jj in [1,2]]+['comp%i_mean_%i'%(comp, ii) for comp in [1,2,3] for ii in [1,2]]+['comp%i_weight'%(comp) for comp in [1,2,3]], data = np.hstack([mc_median, errd, erru, errdd, erruu, std, rot_covariances, rot_means, rot_weights]), index = index)

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

            results = get_statistics(y_pred, params['targets_names'], index = X.index)

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

                results.append(get_statistics(y_pred_i, params['targets_names'], index = X_i.index))

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
        results = get_statistics(y_pred, params['targets_names'], index = X.index)
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


def get_data_test(params, external_cat = None):

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
        if used_photo=='flux_aper':
            photo = [mag for mag in data.columns if 'flux_aper_' in mag]
            photo_err = [flux for flux in data.columns if 'flux_relerr_aper_' in flux]
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
    predict_data = pd.read_csv(params['test_catalog'])

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
        predict_data.loc[:, photo] = np.log10(predict_data.loc[:, photo]+2e3)

        try:
            predict_data.loc[:, [ 'e_fg','e_fbp','e_frp']] = (predict_data.loc[:, [ 'e_fg','e_fbp','e_frp']].values)/((predict_data.loc[:, [ 'fg','fbp','frp']].values)*np.log(10))
            predict_data.loc[:, [ 'fg','fbp','frp']] = np.log10(predict_data.loc[:, [ 'fg','fbp','frp']]+2e3)
        except:
            pass

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

