import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json

# Find current directory
current_dir = os.getcwd()
# Find root directory
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Add root directory to path
sys.path.append(root_dir)

# Import custom modules
import pysindy.pysindy as ps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['method'] = 'RK23'   #'LSODA'
integrator_keywords['rtol'] = 1e-3
integrator_keywords['atol'] = 1e-4

experiment_dir = 'data'
experiment_file = 'Population_Training_Results.dat'
exp_dir = os.path.join(experiment_dir, experiment_file)
system_size_list = [20, 50, 100]
n_samples_train_list = [1000, 5000, 10000]
n_samples_test = 2000
# List of IDs to plot
ids_to_plot = [1, 2, 3, 10, 50, 100]

# Build and fit the model
poly_order_list = [2, 3]
threshold_list = [1e-5, 1e-10]
# dt = 1

def main(system_size, n_samples_train, poly_order, threshold,  save_folder='ck_experiments'):
    
    print('#'*100)
    print(f"Running SINDy for system size {system_size}, n_samples_train {n_samples_train}, poly_order {poly_order}, threshold {threshold}")
    print('#'*100)
    
    # Save json file with the parameters
    params = {
        'poly_order': poly_order,
        'threshold': threshold,
        'system_size': system_size,
        'n_samples_train': n_samples_train,
        'n_samples_test': n_samples_test,
        'ids_to_plot': ids_to_plot,
        'exp_dir': exp_dir,
        'integrator_keywords': integrator_keywords
    }
    # # Save the parameters line by line
    # with open(os.path.join(save_folder, 'params.json'), 'w') as f:
    #     for key, value in params.items():
    #         f.write(f"{json.dumps({key: value})}\n")
    
    # Save the parameters as a single JSON object
    with open(os.path.join(save_folder, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
        
        
    
    print('1. Preparing the data...') 
    # Load experimental data
    t_values, N_clusters = readExpData(exp_dir, Nsize=system_size)
    
    # # n_samples_train random indices
    # indices = np.random.choice(len(t_values), n_samples_train, replace=False)
    # indices = np.sort(indices)
    # Linearly spaced indices
    indices = np.linspace(0, len(t_values)-1, n_samples_train, dtype=int)
    N_clusters_train = N_clusters[indices]
    # t_values_train = t_values[indices]
    # Time step from indices
    dt = t_values[1] - t_values[0]

    print('2. Fitting the model...')
    model = ps.SINDy( 
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    model.fit(N_clusters_train, t=dt)
        
    model.save(save_folder, precision=4)
    
    print('3. Simulating the model...')
    # Simulate the model
    n_sim, t_sim = model.simulate(N_clusters[0], t=t_values[:n_samples_test], integrator_kws=integrator_keywords)
    
    print('4. Plotting the results...')
    # Create subplot grid
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    # Log the time values
    t_values_log = np.log(t_values)
    t_sim_log = np.log(t_sim)

    for i, n_to_plot in enumerate(ids_to_plot):
        
        if n_to_plot > system_size:
            print(f"Cluster size {n_to_plot} is larger than the system size {system_size}")
            continue
        # Plot the data
        axs[i].plot(t_values_log, N_clusters[:, n_to_plot-1], label=f'Experimental data')
        axs[i].plot(t_sim_log, n_sim[:, n_to_plot-1], label='SINDy model')
        axs[i].set_title(f'Cluster size {n_to_plot}')
        axs[i].legend()
        
        axs[i].set_xlabel('log(t)')
        axs[i].set_ylabel('Cluster concentration')

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder,'ck_model_fit.png'))

def readExpData(file_path, Nsize=100):
    
    if os.path.isfile(file_path):
        
        # extract data from file using numpy module
        data = np.loadtxt(file_path)
        size = np.shape(data)

        t_data = np.empty(shape=(size[0],))
        # first column of the file is time
        t_data[:] = data[:, 0]

        # remaining data is number of clusters of size n. Index for column i corresponds to number of clusters of size i-1
        # n_data = data[:, 1:Nsize+1]
        #** We just divide the cluster numbers by the volume of the system which is 126^3 (in units of sigma). 
        n_data = data[:, 1:Nsize+1] / (126**3)  ## only the first 'size' (excluding the very first 0) entries to match the kappa coeff
        
        return t_data, n_data
    
    else:
        print(f"Experiment file ({file_path}) does not exist")
        exit()


if __name__ == '__main__':
    
    # Save the model
    exp_folder = 'ck_experiments'
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    
    for ss in system_size_list:
        for st in n_samples_train_list:
            for po in poly_order_list:
                for th in threshold_list:
                    # Get time in format YYMMDD_HHMMSS
                    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
                    save_folder = os.path.join(exp_folder, f'ck_{current_time}')
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    
                    main(system_size=ss,
                        n_samples_train=st,
                        poly_order=po,
                        threshold=th,
                        save_folder=save_folder
                    )