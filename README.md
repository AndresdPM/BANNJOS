# BANNJOS

**BANNJOS** is a Python-based tool designed for the classification and prediction of astronomical data using machine learning techniques. The code is highly customizable, allowing for various experiment setups, model configurations, and data preprocessing methods. It integrates different libraries to facilitate data manipulation, model training, and prediction.

## Features

- **Customizable Model Architecture**: Define layers, dropout configurations, and loss functions.
- **Data Preprocessing**: Supports variance thresholding, balancing (using SMOTE or undersampling), and feature selection (RFE or ANOVA).
- **Multi-processing Support**: Utilize multiple processors for faster computation.
- **Cross-Validation and Uncertainty Estimation**: Includes cross-validation and variance estimation for robust model performance.
- **Flexible Experimentation**: Supports multiple modes including train, predict, and plot.

## Requirements

- Python 3.6+
- Libraries:
  - `pandas`
  - `polars`
  - `numpy`
  - `sklearn`
  - `imblearn`
  - `argparse`
  - `multiprocessing`
  - Other dependencies like `return_experiment_params`, `libraries` (Ensure these modules are available in your environment)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BANNJOS.git
   cd BANNJOS

2. Install the required packages:

pip install -r requirements.txt


3. Prepare your data and place it in the appropriate directories as expected by the code.

## Installation

# Basic Command

python BANNJOS_train.py --experiment_name "YourExperiment"

# Example Command

python BANNJOS_train.py --experiment_name "Classify" --model_nominal "dropout"
python BANNJOS_train.py --experiment_name "Classify" --model_nominal "deterministic" --layers_nominal 300 100 50

# Main Arguments

- **'--experiment_name'**: Name of the experiment.
- **'--training_catalog'**: Path to the training data file.
- **'--output_path'**: Directory where results will be saved.
- **'--model_nominal'**: Specify the model type (dropout, deterministic).
- **'--layers_nominal'**: Configuration of neural network layers.
- **'--dropout_nominal'**: Dropout rates for the layers.
- **'--batch_size_nominal'**: Batch size for training.
- **'--epochs_nominal'**: Number of training epochs.
- **'--nominal_model_mode'**: Modes include read, train, predict, plot.
- **'--multiprocesing'**: Enable multiprocessing for faster computation.

For a full list of arguments and their descriptions, refer to the code or use the --help option:

python BANNJOS_train.py --help

## Output

- **Results**: Results of predictions, including CSV files with predictions, losses, and optionally PDFs of distributions.
- **Models**: Saved model weights and parameters in the specified output directory. These models can be later used to make large-scale predictions using BANNJOS_predict.py.

## Citation
If you use this code in your research, please cite [del Pino et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240416567D/abstract) where the method is described.

## License

This project is licensed under the MIT License.


