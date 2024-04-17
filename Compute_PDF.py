"""This python code includes all the necessary functions to reconstruct the
3-dimensional probability distribution function derived with BANNJOS. The main
function is "reconstruct_pdf", which should be called as:

pdf = reconstruct_pdf(input_data, features_names, output_dir, rot_mat, n_points = 5000,
                      source_ids = None, idxs = None, save_files = True)

A minimum working example can be found in __main__

Created by Andres del Pino Molina.

"""

import sys
import pandas as pd
import numpy as np

def reconstruct_gmm(rot_covariances, rot_means, rot_weights, rot_mat):
    """Rotate and reconstruct the original GMM parameters given a rotation matrix.

    Inputs
    ----------
        - rot_covariances : Pandas data Series. The rotated and compressed covariance array from BANNJOS.
        - rot_means : Pandas data Series. The rotated and compressed means array from BANNJOS.
        - rot_weights : Pandas data Series. The weights from each one of the Gaussian components produced by BANNJOS.
        - rot_mat : Numpy array 2D array containing the rotation matrix

    Outputs
    ----------
        - covariances_3D : Numpy array. The covariance matrices of the three Gaussian components (3*2*2)
        - means_3D : Numpy array. The means of the three Gaussian components (3*2)
        - weights_3D : Numpy array. The weights of the three Gaussian components (3*1)
    """

    # We count the number of components:
    gmm_n_comp = len(rot_weights)

    # We bring back the covariances and the
    for comp in range(gmm_n_comp):
       rot_covariances['comp%s_cov_21'%(comp+1)] = rot_covariances['comp%s_cov_12'%(comp+1)]

    rot_covariances = rot_covariances.loc[['comp%i_cov_%i%i'%(comp, ii, jj) for comp in [1,2,3] for ii in [1,2] for jj in [1,2]]].values.reshape((3, 2, 2))
    rot_means = rot_means.values.reshape((3, 2))
    rot_weights_3D = rot_weights.values

    # We need now to add an extra column to the covariances
    rot_covariances_3D = np.zeros((3,3,3))
    rot_covariances_3D[:, 0:2, 0:2] =  rot_covariances
    covariances_3D = np.dot(rot_covariances_3D, rot_mat)

    # We create the means arrays
    rot_means_3D = np.zeros((3,3))
    rot_means_3D[:, 0:2] =  rot_means
    means_3D = np.dot(rot_means_3D, rot_mat) + 1/3

    return covariances_3D, means_3D, rot_weights_3D


def sample_from_gmm(covariances, means, weights, n_points = 1000):
    """Randomly sample points from a GMM .

    Inputs
    ----------
        - covariances : Numpy array. The rotated and compressed covariance array from BANNJOS.
        - means : Numpy array. The rotated and compressed means array from BANNJOS.
        - weights : Numpy array. The weights from each one of the Gaussian components produced by BANNJOS.
        - n_points : Integer. Number of points to be sampled.

    Outputs
    ----------
        - gm_data : Numpy array ((n_points, n_dimensions)).
    """

    # We sample the Gaussian in the 3D space.
    gm_data = []
    for weight, mean, covariance in zip(weights, means, covariances):

        # We make sure that the covariance is positive-semidefinite
        min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
        if min_eig < 0:
            covariance -= 10*min_eig * np.eye(*covariance.shape)

        gm_data.append(np.random.multivariate_normal(mean, covariance, size=int(n_points*weight), check_valid='warn', tol=1e-8))

    gm_data = np.concatenate(gm_data)

    return gm_data


def read_data(file_name, source_ids = None, idx = None):
    """Read csv files from BANNJOS.

    Inputs
    ----------
        - file_name : String with the name of a csv file containing a valid BANNJOS output.
        - source_ids : List containing the source ID of the object(s) of interest. The format should be ['88573-6154', '102431-8772', ...]. It can be None for no selection.
        - idx : Index(es) of the object(s) of interest. Can be a list, numpy array, or a generator such as range().

    Outputs
    ----------
        - results : A pandas DataFrame containing the compressed GMM from BANNJOS for the selected objects.
    """

    # Read the data table:
    data = pd.read_csv(file_name)

    if source_ids is not None:

        source_ids = [id.split('-') for id in source_ids]

        results = []
        for tile, number in source_ids:
            results.append(data.loc[(data.tile_id == int(tile)) & (data.number == int(number)), :])
        results = pd.concat(results)

    # If idxs is not defined, run the pdf reconstruction over the entire table.
    elif idx is not None:
        results = data.iloc[idx]

    else:
        idx = list(range(len(data)))
        results = data.iloc[idx]

    return results


def reconstruct_pdf(input_data, features_names, output_dir, rot_mat, n_points = 5000, source_ids = None, idxs = None, save_files = True):
    """Resample the PDF from BANNJOS.

    Inputs
    ----------
        - input_data : The input data. It can be either a pandas DataFrame object, or a string with the name of a csv file containing a valid BANNJOS output.
        - features_names : A list with the names of the final output columns. For example ['a','b','c'].
        - output_dir : A string containing the output directory where to save the re-sampled PDFs.
        - rot_mat : A Numpy array with the rotation matrix needed to reconstruct the results to their original 3D state.
        - n_points : Integer. Number of points to be sampled from the reconstructed GMM.
        - source_ids : List containing the source ID of the object(s) of interest. The format should be ['88573-6154', '102431-8772', ...]. It can be None for no selection.
        - idx : Index(es) of the object(s) of interest. Can be a list, numpy array, or a generator such as range().
        - save_files : Boleean. If True, the PDFs of the individual objects will be saved in a csv file in the output_dir.

    Outputs
    ----------
        - all_PDFs : A list containing the PDFs of all the reconstructed objects.

    """

    # We check the consistency of the input
    if isinstance(input_data, str):
        try:
            # We read the data
            data = read_data(input_data, source_ids = source_ids, idx = idxs)
        except:
            print('We could not read %s. Are you sure it is a csv file?'%input_data)
            print('Quitting now')
            sys.exit()
    elif isinstance(input_data, pd.DataFrame):
        data = input_data
    elif isinstance(input_data, pd.Series):
        data = pd.DataFrame(columns = input_data.index.to_list(),  data = np.expand_dims(input_data.values, 0), index = [1])
    else:
        print('%s seems to be a %s. Please use either the name for a csv file or a Pandas DataFrame.'%(input_data, type(input_data)))
        print('Quitting now')
        sys.exit()

    # We select the columns containing information about the GMM:
    covariances_names = [name for name in data.columns.to_list() if (('comp' in name) and ('cov' in name))]
    means_names = [name for name in data.columns.to_list() if (('comp' in name) and ('mean' in name))]
    weights_names = [name for name in data.columns.to_list() if (('comp' in name) and ('weight' in name))]

    # We create a list where the PDFs will be stored if we prefer not to save the individual files.
    all_PDFs = []

    # We iterate over the sources
    for idx, source in data.iterrows():

        covariances, means, weights = reconstruct_gmm(source.loc[covariances_names], source.loc[means_names], source.loc[weights_names], rot_mat)

        reconstructed_pdf_data = sample_from_gmm(covariances, means, weights, n_points = n_points)

        # We build a DataFrame containing the source PDF
        reconstructed_pdf = pd.DataFrame(data = reconstructed_pdf_data, columns = features_names)

        if save_files:
            # Save the source PDF into a file
            reconstructed_pdf.to_csv('%s/%i_%i_PDF.csv'%(output_dir, source.tile_id, source.number), index = False, float_format = '%.4f')
        else:
            # Append the source PDF to a list of DataFrames.
            all_PDFs.append(reconstructed_pdf)

    return all_PDFs


if __name__ == '__main__':
    """A minimum working example is shown here.
    Inputs
    ----------
        - input_data : The file name containing an output table from BANNJOS.
        - output_dir : Directory where the reconstructed PDFs will be saved.
        - features_names : Names of the saved columns in the reconstructed PDF.
        - rot_mat : A Numpy array with the rotation matrix needed to reconstruct the results to their original 3D state.
        - n_points : Integer. Number of points to be sampled from the reconstructed GMM.
        - source_ids : List containing the source ID of the object(s) of interest. The format should be ['88573-6154', '102431-8772', ...]. It can be None for no selection.
        - idx : Index(es) of the object(s) of interest. Can be a list, numpy array, or a generator such as range().
        - save_files : Boleean. If True, the PDFs of the individual objects will be saved in a csv file in the output_dir.

    Outputs
    ----------
        - pdf : Reconstructed PDF as a pandas DataFrame.
    """

    # This is the input file name

    input_data = 'J-PLUS_DR3_85409_class.csv'
    output_dir = './reconstructed_pdfs'

    features_names = ['P_Galaxy', 'P_QSO', 'P_Star']

    # We define the rotation matrix here
    rot_mat = np.array([[(np.sqrt(3)+3)/6,           -np.sqrt((2-np.sqrt(3))/6), -1/np.sqrt(3)],
                        [-np.sqrt((2-np.sqrt(3))/6), (np.sqrt(3)+3)/6,           -1/np.sqrt(3)],
                        [1/np.sqrt(3),                1/np.sqrt(3),               1/np.sqrt(3)]])

    # We define how many points we want when sampling the PDF
    n_points = 3000

    # Save individual PDFs
    save_files = True

    # Here, we can request for specific objects ids in the data table. We can do that by specifying the tile and the number of the object in two different lists. You can specify several objects.
    source_ids = None
    #source_ids = ['88573-6154', '102431-8772']

    # Another option is to specify the index in the data table. The index (idx) must coincide with the row number in the data table. Comment if you want all the sources in data.
    idxs = range(10)
    #idxs = [300]

    # Here we call the script that computes the PDF
    pdf = reconstruct_pdf(input_data, features_names, output_dir, rot_mat, n_points = n_points, source_ids = source_ids, idxs = idxs, save_files = save_files)

    # Users can introduce more code here in case they want to use all_PDFs directly.
