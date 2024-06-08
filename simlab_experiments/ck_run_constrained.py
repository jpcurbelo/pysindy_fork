import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from multiprocessing import Pool
import warnings
from contextlib import contextmanager
from copy import copy
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from pathlib import Path

# Find current directory
current_dir = os.getcwd()
# Find root directory
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add root directory to path
sys.path.append(root_dir)

# Import custom modules
import pysindy.pysindy as ps
from pysindy.differentiation import SmoothedFiniteDifference

from ck_utils import (
    generate_random_word, 
    ck_constraints, 
    extract_and_save_k_values,
    readCKExpData,
)

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

system_size_list = [50]
# system_size_list = [5, 10, 20]
n_samples_train_list = [1000, 5000, 10000, 16000]
regularizer_list = ['l1', 'l2']
poly_order = 2

experiment_dir = 'data'
experiment_file = 'Population_Training_Results.dat'
exp_dir = os.path.join(experiment_dir, experiment_file)

num_workers = 2
exp_folder = 'ck_experiments_constrained'

def main(system_size, n_samples_train, regularizer, save_folder='ck_experiments'):

    params = {
        'system_size': system_size,
        'n_samples_train': n_samples_train,
        'regularizer': regularizer,
    }

    # Save the parameters as a single JSON object
    with open(os.path.join(save_folder, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    ## Load experimental data
    t_values, N_clusters = readCKExpData(exp_dir, Nsize=system_size)

    # Linearly spaced indices
    indices = np.linspace(0, len(t_values)-1, n_samples_train, dtype=int)
    N_clusters_train = N_clusters[indices]
    # Time step from indices
    dt = t_values[1] - t_values[0]

    ## Prepare the model (add constraints)
    # Figure out how many library features there will be
    library = ps.PolynomialLibrary()
    library.fit([ps.AxesArray(N_clusters_train, {"ax_sample": 0, "ax_coord": 1})])
    feature_names = library.get_feature_names()

    ## Set inequalities
    # Repeat with inequality constraints, need CVXPY installed
    try:
        import cvxpy  # noqa: F401
        run_cvxpy = True
    except ImportError:
        run_cvxpy = False
        print("No CVXPY package is installed")

    print("Starting inequality constraints...", save_folder)
    constraint_lhs, constraint_rhs = ck_constraints(N_clusters_train, feature_names)
    print("Done building inequality constraints...", save_folder)

    ## Build and fit the model
    smoothed_fd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
    differentiation_method = smoothed_fd

    optimizer = ps.ConstrainedSR3(
        # verbose=True,
        constraint_rhs=constraint_rhs,
        constraint_lhs=constraint_lhs,
        inequality_constraints=True,  # Ensure this is True for inequality constraints
        thresholder=regularizer,
        tol=1e-12,
        threshold=1e-12,
        max_iter=100000,
    )

    feature_library = ps.PolynomialLibrary(degree=poly_order)

    with ignore_specific_warnings():
        # Fit the model
        model = ps.SINDy( 
            # discrete_time=True,
            differentiation_method=differentiation_method,
            optimizer=optimizer,
            feature_library=feature_library,
        )
        model.fit(N_clusters_train, t=dt)

    print("Model fitted...", save_folder)

    ## Save the model
    model.save(save_folder, precision=6)

    ## Extract and save k values
    extract_and_save_k_values(feature_names, optimizer, Path(save_folder) / 'kappa_values.csv')


@contextmanager
def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    warnings.filters = filters

def process_params(params):
     
    ss, nst, reg = params
    current_time = datetime.now().strftime('%m%d_%H%M%S')
    rand_name = generate_random_word()
    save_folder = os.path.join(exp_folder, f'ck_{ss}clusters_{current_time}_{rand_name}')
    os.makedirs(save_folder)

    main(system_size=ss, 
         n_samples_train=nst,
         regularizer=reg,
         save_folder=save_folder
    )

    
if __name__ == '__main__':
    
    # Save the model
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    # Create a list of parameter combinations
    parameter_combinations = [(ss, nst, reg) for ss in system_size_list 
                                        for nst in n_samples_train_list
                                        for reg in regularizer_list
                            ]

    # Create a Pool of worker processes
    with Pool(num_workers) as pool:
        # Execute the processes in parallel
        pool.map(process_params, parameter_combinations) 