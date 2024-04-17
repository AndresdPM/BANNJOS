
def return_experiment_params(args):

    if args.experiment_name is None:
        args.experiment_name = '%s_%s_%s_layers_%s'%(args.model_nominal, args.loss_nominal, args.batch_size_nominal, '_'.join([str(int) for int in args.layers_nominal]))

    variables_n_preproc = {'training_catalog': args.training_catalog,
                           'output_path': args.output_path,
                           'training_flags': args.training_flags,
                           'test_catalog': None,
                           'y_vars':args.y_vars,
                           'y_vars_err':args.y_vars_err,
                           'used_photo': 'flux_aper', #, 'flux_psfcor', 'mag_aper', 'mag_3_worstpsf', 'flux_psfcor'
                           'use_photo_error':True,
                           'use_gal_lat':False,
                           'each_nrows': args.each_nrows,
                           'skip_nrows': args.skip_nrows,
                           'nrows': args.nrows,
                           'validation_sample_size':0.10,
                           'save_validation_catalog': True,
                           'save_training_catalog': args.save_training_catalog,
                           'random_generator_seed':42,
                           'y_drop_nans': True,
                           'select_class': None,
                           'photo_log':True,
                           'stratify_var':None,
                           'fill_na': -0.1}

    independent_vec, quality_vars, validation_vars, prediction_vars = return_vectors(variables_n_preproc['use_photo_error'])

    variables_n_preproc.update({'X_vars': independent_vec,
                                'quality_vars':quality_vars,
                                'validation_vars': validation_vars,
                                'prediction_vars':prediction_vars
                                })

    model_par_nominal = {'model': args.model_nominal,
                        'hidden_layers': args.layers_nominal,
                        'hidden_dropout': args.dropout_nominal,
                        'input_dropout': args.input_dropout_nominal,
                        'loss':args.loss_nominal, #'mean_squared_error' # mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, huber, log_cosh
                        'mode':args.nominal_model_mode,
                        'epochs':args.epochs_nominal,
                        'batch_size': args.batch_size_nominal,
                        'initial_learning_rate': args.initial_learning_rate,
                        'step_decay_learning_rate': args.step_decay_learning_rate,
                        'final_learning_rate': 1e-7,
                        'early_stop_patience':args.early_stop_patience_nominal,
                        'delta_early_stop_patience':1e-6,
                        'epistemic_bootstrap': False,
                        'epistemic_montecarlo': False,
                        'alleatoric_montecarlo':False,
                        'epistemic_n_iter':1,
                        'alleatoric_n_iter':args.nominal_alleatoric_n_iter,
                        'full_cov':args.full_cov,
                        'verbose':args.verbose,
                        'pre_trained_weights_file': '%s/model/weights_nominal.hdf5'%(variables_n_preproc['output_path']+args.experiment_name)
                        }

    model_par_variance = {'model': 'deterministic',
                        'hidden_layers': [128, 128],
                        'hidden_dropout': [0., 0., 0., 0.],
                        'input_dropout': 0,
                        'loss':'mean_squared_error',
                        'mode':'train',
                        'epochs':2000,
                        'batch_size': 64,
                        'initial_learning_rate': 1e-4,
                        'step_decay_learning_rate': 25,
                        'final_learning_rate': 1e-7,
                        'early_stop_patience':200,
                        'epistemic_bootstrap': False,
                        'epistemic_montecarlo': False,
                        'alleatoric_montecarlo':False,
                        'epistemic_n_iter':1,
                        'alleatoric_n_iter':1,
                        'verbose':args.verbose,
                        'pre_trained_weights_file': '%s/model/weights_variance.hdf5'%(variables_n_preproc['output_path']+args.experiment_name)
                        }


    experiment = {'perform_RFE':False,
                  'perform_ANOVA':False,
                  'Balance_training':'SMOTE',
                  'n_chunks':args.n_chunks,
                  'multiprocesing':args.multiprocesing,
                  'n_processors':args.n_processors,
                  'fit_uncertainty_cv':False,
                  'perform_cv':True,
                  'nsplits_cv':8,
                  'nrepeats_cv':1
                  }

    params = {'experiment_name': args.experiment_name,
            'experiment':experiment,
            'variables_n_preproc':variables_n_preproc,
            'model_nominal':model_par_nominal,
            'model_variance':model_par_variance}

    return params



def return_vectors(use_error_in_fit):

    if use_error_in_fit:

        data_quality = ['J0430_ZPT','J0378_ZPT', 'J0410_ZPT', 'gSDSS_ZPT', 'J0660_ZPT', 'J0395_ZPT', 'zSDSS_ZPT', 'rSDSS_ZPT', 'iSDSS_ZPT', 'uJAVA_ZPT', 'J0861_ZPT', 'J0515_ZPT', 'J0430_NOISE', 'J0378_NOISE', 'J0410_NOISE', 'gSDSS_NOISE', 'J0660_NOISE', 'J0395_NOISE', 'zSDSS_NOISE', 'rSDSS_NOISE', 'iSDSS_NOISE', 'uJAVA_NOISE', 'J0861_NOISE', 'J0515_NOISE', 'J0430_NOISE_RMS', 'J0378_NOISE_RMS', 'J0410_NOISE_RMS', 'gSDSS_NOISE_RMS', 'J0660_NOISE_RMS', 'J0395_NOISE_RMS', 'zSDSS_NOISE_RMS', 'rSDSS_NOISE_RMS', 'iSDSS_NOISE_RMS', 'uJAVA_NOISE_RMS', 'J0861_NOISE_RMS', 'J0515_NOISE_RMS', 'J0430_ELLIPTICITY_MEAN', 'J0378_ELLIPTICITY_MEAN', 'J0410_ELLIPTICITY_MEAN', 'gSDSS_ELLIPTICITY_MEAN', 'J0660_ELLIPTICITY_MEAN', 'J0395_ELLIPTICITY_MEAN', 'zSDSS_ELLIPTICITY_MEAN', 'rSDSS_ELLIPTICITY_MEAN', 'iSDSS_ELLIPTICITY_MEAN', 'uJAVA_ELLIPTICITY_MEAN', 'J0861_ELLIPTICITY_MEAN', 'J0515_ELLIPTICITY_MEAN', 'J0430_FWHM_MEAN', 'J0378_FWHM_MEAN', 'J0410_FWHM_MEAN', 'gSDSS_FWHM_MEAN', 'J0660_FWHM_MEAN', 'J0395_FWHM_MEAN', 'zSDSS_FWHM_MEAN', 'rSDSS_FWHM_MEAN', 'iSDSS_FWHM_MEAN', 'uJAVA_FWHM_MEAN', 'J0861_FWHM_MEAN', 'J0515_FWHM_MEAN', 'J0430_FWHMG', 'J0378_FWHMG', 'J0410_FWHMG', 'gSDSS_FWHMG', 'J0660_FWHMG', 'J0395_FWHMG', 'zSDSS_FWHMG', 'rSDSS_FWHMG', 'iSDSS_FWHMG', 'uJAVA_FWHMG', 'J0861_FWHMG', 'J0515_FWHMG', 'J0430_FWHMG_RMS', 'J0378_FWHMG_RMS', 'J0410_FWHMG_RMS', 'gSDSS_FWHMG_RMS', 'J0660_FWHMG_RMS', 'J0395_FWHMG_RMS', 'zSDSS_FWHMG_RMS', 'rSDSS_FWHMG_RMS', 'iSDSS_FWHMG_RMS', 'uJAVA_FWHMG_RMS', 'J0861_FWHMG_RMS', 'J0515_FWHMG_RMS', 'J0430_MOFFATBETA_MEAN', 'J0378_MOFFATBETA_MEAN', 'J0410_MOFFATBETA_MEAN', 'gSDSS_MOFFATBETA_MEAN', 'J0660_MOFFATBETA_MEAN', 'J0395_MOFFATBETA_MEAN', 'zSDSS_MOFFATBETA_MEAN', 'rSDSS_MOFFATBETA_MEAN', 'iSDSS_MOFFATBETA_MEAN', 'uJAVA_MOFFATBETA_MEAN', 'J0861_MOFFATBETA_MEAN', 'J0515_MOFFATBETA_MEAN', 'J0430_DEPTH2FWHM5S', 'J0378_DEPTH2FWHM5S', 'J0410_DEPTH2FWHM5S', 'gSDSS_DEPTH2FWHM5S', 'J0660_DEPTH2FWHM5S', 'J0395_DEPTH2FWHM5S', 'zSDSS_DEPTH2FWHM5S', 'rSDSS_DEPTH2FWHM5S', 'iSDSS_DEPTH2FWHM5S', 'uJAVA_DEPTH2FWHM5S', 'J0861_DEPTH2FWHM5S', 'J0515_DEPTH2FWHM5S', 'J0430_DEPTH3ARC5S', 'J0378_DEPTH3ARC5S', 'J0410_DEPTH3ARC5S', 'gSDSS_DEPTH3ARC5S', 'J0660_DEPTH3ARC5S', 'J0395_DEPTH3ARC5S', 'zSDSS_DEPTH3ARC5S', 'rSDSS_DEPTH3ARC5S', 'iSDSS_DEPTH3ARC5S', 'uJAVA_DEPTH3ARC5S', 'J0861_DEPTH3ARC5S', 'J0515_DEPTH3ARC5S', 'J0430_M50S', 'J0378_M50S', 'J0410_M50S', 'gSDSS_M50S', 'J0660_M50S', 'J0395_M50S', 'zSDSS_M50S', 'rSDSS_M50S', 'iSDSS_M50S', 'uJAVA_M50S', 'J0861_M50S', 'J0515_M50S', 'J0430_KS', 'J0378_KS', 'J0410_KS', 'gSDSS_KS', 'J0660_KS', 'J0395_KS', 'zSDSS_KS', 'rSDSS_KS', 'iSDSS_KS', 'uJAVA_KS', 'J0861_KS', 'J0515_KS', 'J0430_M50G', 'J0378_M50G', 'J0410_M50G', 'gSDSS_M50G', 'J0660_M50G', 'J0395_M50G', 'zSDSS_M50G', 'rSDSS_M50G', 'iSDSS_M50G', 'uJAVA_M50G', 'J0861_M50G', 'J0515_M50G', 'J0430_KG', 'J0378_KG', 'J0410_KG', 'gSDSS_KG', 'J0660_KG', 'J0395_KG', 'zSDSS_KG', 'rSDSS_KG', 'iSDSS_KG', 'uJAVA_KG', 'J0861_KG', 'J0515_KG', 'flags_uJAVAB', 'flags_J0378B', 'flags_J0395B','flags_J0410B', 'flags_J0430B', 'flags_gSDSSB', 'flags_J0515B', 'flags_rSDSSB', 'flags_J0660B',   'flags_iSDSSB', 'flags_J0861B', 'flags_zSDSSB', 'norm_wmap_val_uJAVAB',   'norm_wmap_val_J0378B', 'norm_wmap_val_J0395B', 'norm_wmap_val_J0410B',   'norm_wmap_val_J0430B', 'norm_wmap_val_gSDSSB', 'norm_wmap_val_J0515B',   'norm_wmap_val_rSDSSB', 'norm_wmap_val_J0660B', 'norm_wmap_val_iSDSSB',   'norm_wmap_val_J0861B', 'norm_wmap_val_zSDSSB', 'mask_flags_uJAVAB', 'mask_flags_J0378B', 'mask_flags_J0395B', 'mask_flags_J0410B', 'mask_flags_J0430B', 'mask_flags_gSDSSB', 'mask_flags_J0515B', 'mask_flags_rSDSSB', 'mask_flags_J0660B', 'mask_flags_iSDSSB', 'mask_flags_J0861B', 'mask_flags_zSDSSB']

        Gaia_vars = ['pmra_g', 'pmde_g', 'plx', 'ruwe', 'fg', 'fbp', 'frp', 'e_pmra_g', 'e_pmde_g', 'e_plx', 'e_fg', 'e_fbp', 'e_frp', 'angdist_gaia']
        external_photo = ['pmra_cw', 'e_pmra_cw', 'pmde_cw', 'e_pmde_cw', 'w1mpropm_cw', 'e_w1mpropm_cw', 'w2mpropm_cw', 'e_w2mpropm_cw', 'angdist_cw', 'jmag', 'hmag', 'kmag', 'e_jmag', 'e_hmag', 'e_kmag', 'angdist_aw']

    else:
        data_quality = ['J0430_NOISE', 'J0378_NOISE', 'J0410_NOISE', 'gSDSS_NOISE', 'J0660_NOISE', 'J0395_NOISE', 'zSDSS_NOISE', 'rSDSS_NOISE', 'iSDSS_NOISE', 'uJAVA_NOISE', 'J0861_NOISE', 'J0515_NOISE', 'J0430_ELLIPTICITY_MEAN', 'J0378_ELLIPTICITY_MEAN', 'J0410_ELLIPTICITY_MEAN', 'gSDSS_ELLIPTICITY_MEAN', 'J0660_ELLIPTICITY_MEAN', 'J0395_ELLIPTICITY_MEAN', 'zSDSS_ELLIPTICITY_MEAN', 'rSDSS_ELLIPTICITY_MEAN', 'iSDSS_ELLIPTICITY_MEAN', 'uJAVA_ELLIPTICITY_MEAN', 'J0861_ELLIPTICITY_MEAN', 'J0515_ELLIPTICITY_MEAN', 'J0430_FWHM_MEAN', 'J0378_FWHM_MEAN', 'J0410_FWHM_MEAN', 'gSDSS_FWHM_MEAN', 'J0660_FWHM_MEAN', 'J0395_FWHM_MEAN', 'zSDSS_FWHM_MEAN', 'rSDSS_FWHM_MEAN', 'iSDSS_FWHM_MEAN', 'uJAVA_FWHM_MEAN', 'J0861_FWHM_MEAN', 'J0515_FWHM_MEAN', 'J0430_FWHMG', 'J0378_FWHMG', 'J0410_FWHMG', 'gSDSS_FWHMG', 'J0660_FWHMG', 'J0395_FWHMG', 'zSDSS_FWHMG', 'rSDSS_FWHMG', 'iSDSS_FWHMG', 'uJAVA_FWHMG', 'J0861_FWHMG', 'J0515_FWHMG', 'J0430_MOFFATBETA_MEAN', 'J0378_MOFFATBETA_MEAN', 'J0410_MOFFATBETA_MEAN', 'gSDSS_MOFFATBETA_MEAN', 'J0660_MOFFATBETA_MEAN', 'J0395_MOFFATBETA_MEAN', 'zSDSS_MOFFATBETA_MEAN', 'rSDSS_MOFFATBETA_MEAN', 'iSDSS_MOFFATBETA_MEAN', 'uJAVA_MOFFATBETA_MEAN', 'J0861_MOFFATBETA_MEAN', 'J0515_MOFFATBETA_MEAN', 'flags_uJAVAB', 'flags_J0378B', 'flags_J0395B','flags_J0410B', 'flags_J0430B', 'flags_gSDSSB', 'flags_J0515B', 'flags_rSDSSB', 'flags_J0660B',   'flags_iSDSSB', 'flags_J0861B', 'flags_zSDSSB', 'norm_wmap_val_uJAVAB',   'norm_wmap_val_J0378B', 'norm_wmap_val_J0395B', 'norm_wmap_val_J0410B',   'norm_wmap_val_J0430B', 'norm_wmap_val_gSDSSB', 'norm_wmap_val_J0515B',   'norm_wmap_val_rSDSSB', 'norm_wmap_val_J0660B', 'norm_wmap_val_iSDSSB',   'norm_wmap_val_J0861B', 'norm_wmap_val_zSDSSB', 'mask_flags_uJAVAB', 'mask_flags_J0378B', 'mask_flags_J0395B', 'mask_flags_J0410B', 'mask_flags_J0430B', 'mask_flags_gSDSSB', 'mask_flags_J0515B', 'mask_flags_rSDSSB', 'mask_flags_J0660B', 'mask_flags_iSDSSB', 'mask_flags_J0861B', 'mask_flags_zSDSSB']

        Gaia_vars = ['pmra_g', 'pmde_g', 'plx', 'ruwe', 'fg', 'fbp', 'frp']
        external_photo = ['pmra_cw', 'pmde_cw', 'w1mpropm_cw', 'w2mpropm_cw', 'angdist_cw', 'jmag', 'hmag', 'kmag']

    extra_photo_vars = ['mu_max_uJAVAB', 'mu_max_J0378B', 'mu_max_J0395B', 'mu_max_J0410B', 'mu_max_J0430B', 'mu_max_gSDSSB', 'mu_max_J0515B', 'mu_max_rSDSSB', 'mu_max_J0660B', 'mu_max_iSDSSB', 'mu_max_J0861B', 'mu_max_zSDSSB']

    Stx_vars = ['x_image', 'y_image', 'r_eff','fwhm_world', 'a_world', 'b_world','theta_j2000', 'isoarea_world', 'petro_radius', 'kron_radius']

    Reddening_vars = ['ebv']

    training_quality = ['CLASS']

    independent_vec = data_quality+extra_photo_vars+Gaia_vars+Stx_vars+external_photo+Reddening_vars

    validation_vars = ['tile_id', 'number', 'alpha_j2000', 'delta_j2000', 'class_star', 'plx', 'pmra_g', 'pmde_g', 'magab_mag_auto_uJAVAB', 'magab_mag_auto_J0378B', 'magab_mag_auto_J0395B', 'magab_mag_auto_J0410B', 'magab_mag_auto_J0430B', 'magab_mag_auto_gSDSSB', 'magab_mag_auto_J0515B', 'magab_mag_auto_rSDSSB', 'magab_mag_auto_J0660B', 'magab_mag_auto_iSDSSB', 'magab_mag_auto_J0861B', 'magab_mag_auto_zSDSSB', 'magab_mag_err_auto_uJAVAB', 'magab_mag_err_auto_J0378B', 'magab_mag_err_auto_J0395B', 'magab_mag_err_auto_J0410B', 'magab_mag_err_auto_J0430B', 'magab_mag_err_auto_gSDSSB', 'magab_mag_err_auto_J0515B', 'magab_mag_err_auto_rSDSSB', 'magab_mag_err_auto_J0660B', 'magab_mag_err_auto_iSDSSB', 'magab_mag_err_auto_J0861B', 'magab_mag_err_auto_zSDSSB', 'ebv', 'ebv_err'] + training_quality + external_photo

    prediction_vars = ['tile_id', 'number', 'alpha_j2000', 'delta_j2000', 'class_star', 'plx', 'pmra_g', 'pmde_g', 'magab_mag_auto_uJAVAB', 'magab_mag_auto_J0378B', 'magab_mag_auto_J0395B', 'magab_mag_auto_J0410B', 'magab_mag_auto_J0430B', 'magab_mag_auto_gSDSSB', 'magab_mag_auto_J0515B', 'magab_mag_auto_rSDSSB', 'magab_mag_auto_J0660B', 'magab_mag_auto_iSDSSB', 'magab_mag_auto_J0861B', 'magab_mag_auto_zSDSSB', 'magab_mag_err_auto_uJAVAB', 'magab_mag_err_auto_J0378B', 'magab_mag_err_auto_J0395B', 'magab_mag_err_auto_J0410B', 'magab_mag_err_auto_J0430B', 'magab_mag_err_auto_gSDSSB', 'magab_mag_err_auto_J0515B', 'magab_mag_err_auto_rSDSSB', 'magab_mag_err_auto_J0660B', 'magab_mag_err_auto_iSDSSB', 'magab_mag_err_auto_J0861B', 'magab_mag_err_auto_zSDSSB', 'ebv', 'ebv_err'] + external_photo

    return independent_vec, training_quality, validation_vars, prediction_vars


