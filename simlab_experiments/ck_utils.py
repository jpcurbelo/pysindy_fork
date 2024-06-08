import random
from nltk.corpus import words
import numpy as np
import os
import csv
from pathlib import Path
from scipy import integrate
import numba as nb
from numbalsoda import (
    solve_ivp, 
    lsoda_sig,
    lsoda, 
    address_as_void_pointer
)
from numba import types

eps_value = 1e-10

def generate_random_word():
    word_list = words.words()
    filtered_words = [word.lower() for word in word_list if len(word) == 4]
    return random.choice(filtered_words)

def readCKExpData(file_path, Nsize=100):
    
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

def ck_constraints(clusters, feature_names):

    n_targets = clusters.shape[1]
    n_features = len(feature_names)

    ############################################################
    ### Set up and 1st equation
    ############################################################
    ## Coefficients of quadratic terms are zero except for x0^2
    # Find indices of quadratic terms excluding 'x0^2'
    quadratic_idx_first = [i for i, feature_name in enumerate(feature_names) \
                        if ('^2' in feature_name and 'x0' not in feature_name) \
                            or (feature_name.count('x') == 2 and 'x0' not in feature_name)]
    n_quadratic_first = len(quadratic_idx_first)

    # Initialize constraint_rhs and constraint_lhs for quadratic terms
    constraint_rhs = np.full((2 * n_quadratic_first, ), eps_value)
    constraint_lhs = np.zeros((2 * n_quadratic_first, n_targets * n_features))

    # Set coefficients of each quadratic term (excluding 'x0^2') to be zero -> (>= -eps and <= eps)
    for i, idx in enumerate(quadratic_idx_first):
        constraint_lhs[2 * i, idx] = 1
        constraint_lhs[2 * i + 1, idx] = -1

    ## Coefficients of quadratic terms are zero except for x0^2
    # Find indices of quadratic terms excluding 'x0^2'
    quadratic_idx_first = [i for i, feature_name in enumerate(feature_names) \
                        if ('^2' in feature_name and 'x0' not in feature_name) \
                            or (feature_name.count('x') == 2 and 'x0' not in feature_name)]
    n_quadratic_first = len(quadratic_idx_first)

    # Initialize constraint_rhs and constraint_lhs for quadratic terms
    constraint_rhs = np.append(constraint_rhs, np.full((2 * n_quadratic_first, ), eps_value))
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((2 * n_quadratic_first, n_targets * n_features))))

    # Set coefficients of each quadratic term (excluding 'x0^2') to be zero -> (>= -eps and <= eps)
    for i, idx in enumerate(quadratic_idx_first):
        constraint_lhs[-2 * n_quadratic_first + 2 * i, idx] = 1
        constraint_lhs[-2 * n_quadratic_first + 2 * i + 1, idx] = -1

    ## Coefficient of x0^2 is positive
    # Find the index of the x0^2 term
    x0sqs_idx = feature_names.index('x0^2')
    constraint_rhs = np.append(constraint_rhs, eps_value)
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
    constraint_lhs[-1, x0sqs_idx] = -1

    ## Coefficient of x0xN is negative
    # Find the index of the x0xN term
    x0xN_idx = [i for i, feature_name in enumerate(feature_names) if ('x0' in feature_name) \
                and (feature_name.count('x')) == 2]
    for idx in x0xN_idx:
        constraint_rhs = np.append(constraint_rhs, eps_value)
        constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
        constraint_lhs[-1, idx] = 1

    ## Coefficient of linear terms (except for 0) are positive
    # Find the indices of linear terms (excluding 'x0')
    linear_idx = [i for i, feature_name in enumerate(feature_names) if ('x0' not in feature_name) \
                and (feature_name.count('x')) == 1]
    for idx in linear_idx:
        constraint_rhs = np.append(constraint_rhs, eps_value)
        constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
        constraint_lhs[-1, idx] = -1

    ## Coefficient of x0 is zero
    constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
    x0_idx = feature_names.index('x0')
    constraint_lhs[-2, x0_idx] = 1
    constraint_lhs[-1, x0_idx] = -1

    ## Constant term is zero -> (>= -eps and <= eps)
    # Find the index of the constant term '1'
    constant_term_idx = feature_names.index('1')
    constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
    constraint_lhs[-2, constant_term_idx] = 1
    constraint_lhs[-1, constant_term_idx] = -1

    ## Ensure x0^2 coefficient is positive
    # x0^2 >= eps
    constraint_rhs = np.append(constraint_rhs, eps_value)
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
    x0sqs_idx = feature_names.index('x0^2')
    constraint_lhs[-1, x0sqs_idx] = -1

    ############################################################
    ### Eq 2 to N-1 equations
    ############################################################

    ### Add constraints for the middle cluster sizes (Eq 2 to N-1)
    for eq in range(1, n_targets - 1):
        ## All linear terms except for xN and xN+1  to be zero -> (>= -eps and <= eps)
        # Find the indices of linear terms excluding 'xN' and 'xN+1'
        linear_idx = [i for i, feature_name in enumerate(feature_names) if 
                        (feature_name.count('x') == 1 and f'x{eq}' != feature_name and f'x{eq + 1}' != feature_name) 
                        and '^' not in feature_name]

        n_linear = len(linear_idx)

        # Append to constraint_rhs and constraint_lhs
        last_row = constraint_lhs.shape[0]
        constraint_rhs = np.append(constraint_rhs, [eps_value] * 2 * n_linear)
        constraint_lhs = np.vstack((constraint_lhs, np.zeros((2 * n_linear, n_targets * n_features))))

        # Set coefficients of each linear term (excluding 'xN' and 'xN+1') to be zero -> (>= -eps and <= eps)
        for i, idx in enumerate(linear_idx):
            constraint_lhs[2 * i + last_row, eq * n_features + idx] = 1
            constraint_lhs[2 * i + 1 + last_row, eq * n_features + idx] = -1

        ## Coefficient of xN is negative
        # Find the index of the xN term
        xN_idx = feature_names.index(f'x{eq}')
        constraint_rhs = np.append(constraint_rhs, eps_value)
        constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
        constraint_lhs[-1, eq * n_features + xN_idx] = 1

        if eq > 1:
            ## Coefficient of xN is in Eq eq and Eq1 are to be negatives of one another 
            constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
            constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
            constraint_lhs[-2, eq * n_features + xN_idx] = 1
            constraint_lhs[-2, xN_idx] = 1
            constraint_lhs[-1, eq * n_features + xN_idx] = -1
            constraint_lhs[-1, xN_idx] = -1
        else:
            ## Coefficient of xN is in Eq eq is to be half the value of Eq 1
            constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
            constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
            constraint_lhs[-2, eq * n_features + xN_idx] = 1
            constraint_lhs[-2, xN_idx] = 0.5
            constraint_lhs[-1, eq * n_features + xN_idx] = -1
            constraint_lhs[-1, xN_idx] = -0.5

        ## Coefficient of xN+1 is positive
        # Find the index of the xN+1 term
        xN1_idx = feature_names.index(f'x{eq + 1}')
        constraint_rhs = np.append(constraint_rhs, eps_value)
        constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
        constraint_lhs[-1, eq * n_features + xN1_idx] = -1

        ## Coefficient of xN+1 is in Eq eq and Eq1 are to be equal
        constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
        constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
        constraint_lhs[-2, eq * n_features + xN1_idx] = 1
        constraint_lhs[-2, xN1_idx] = -1
        constraint_lhs[-1, eq * n_features + xN1_idx] = -1
        constraint_lhs[-1, xN1_idx] = 1

        ## Constant term is zero -> (>= -eps and <= eps)
        # Find the index of the constant term '1'
        constant_term_idx = feature_names.index('1')
        constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
        constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
        constraint_lhs[-2, eq * n_features + constant_term_idx] = 1
        constraint_lhs[-1, eq * n_features + constant_term_idx] = -1

        ## All quadratic terms except for x0 xN and x0 xN-1 (including x0^2) to be zero -> (>= -eps and <= eps)
        # Find indices of quadratic terms excluding 'x0 xN' and 'x0 xN-1'
        quadratic_idx_mix = [i for i, feature_name in enumerate(feature_names) if 
                                ((feature_name.count('x') == 2 and f'x0 x{eq}' != feature_name and f'x0 x{eq - 1}' != feature_name) 
                                or ('^2' in feature_name and 'x0' not in feature_name))
                                or ('x0^2' in feature_name and eq != 1)]
        n_quadratic_mix = len(quadratic_idx_mix)

        # Add constraints for quadratic terms
        last_row = constraint_lhs.shape[0]
        constraint_rhs = np.append(constraint_rhs, [eps_value] * 2 * n_quadratic_mix)
        constraint_lhs = np.vstack((constraint_lhs, np.zeros((2 * n_quadratic_mix, n_targets * n_features))))

        for i, idx in enumerate(quadratic_idx_mix):
            constraint_lhs[2 * i + last_row, eq * n_features + idx] = 1
            constraint_lhs[2 * i + 1 + last_row, eq * n_features + idx] = -1

        ## Coefficient of x0 xN is negative
        # Find the index of the x0 xN term
        x0xN_idx = [i for i, feature_name in enumerate(feature_names) if ('x0' in feature_name) and (f'x0 x{eq}' in feature_name)]
        for idx in x0xN_idx:
            constraint_rhs = np.append(constraint_rhs, eps_value)
            constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
            constraint_lhs[-1, eq * n_features + idx] = 1

        ## Coefficient of x0 xN in Eq eq and Eq1 are to be equal
        constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
        constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
        constraint_lhs[-2, eq * n_features + idx] = 1
        constraint_lhs[-2, idx] = -1
        constraint_lhs[-1, eq * n_features + idx] = -1
        constraint_lhs[-1, idx] = 1

        ## Coefficient of x0 xN-1 is positive
        # Find the index of the x0 xN-1 term
        x0xN1_idx = [i for i, feature_name in enumerate(feature_names) if 
                    ('x0^2' in feature_name) or (f'x0 x{eq - 1}' in feature_name)]
        for idx in x0xN1_idx:
            constraint_rhs = np.append(constraint_rhs, eps_value)
            constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
            constraint_lhs[-1, eq * n_features + idx] = -1

        if eq > 1:
            ## Coefficient of x0 xN-1 in Eq eq and Eq1 are to be negatives of one another
            constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
            constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
            constraint_lhs[-2, eq * n_features + idx] = 1
            constraint_lhs[-2, idx] = 1
            constraint_lhs[-1, eq * n_features + idx] = -1
            constraint_lhs[-1, idx] = -1
        else:
            ## Coefficient of x0 xN-1 in Eq eq is to be half the value of Eq 1
            constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
            constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
            constraint_lhs[-2, eq * n_features + idx] = 1
            constraint_lhs[-2, idx] = -0.5
            constraint_lhs[-1, eq * n_features + idx] = -1
            constraint_lhs[-1, idx] = 0.5

    ############################################################
    ### Last equation
    ############################################################

    ### Add constraints for the last cluster size (Eq N)
    ## All linear terms to be zero except for xN
    linear_idx_last = [i for i, feature_name in enumerate(feature_names) if feature_name.count('x') == 1 
                    and '^' not in feature_name 
                    and f'x{n_targets - 1}' not in feature_name]
    n_linear = len(linear_idx_last)

    last_row = constraint_lhs.shape[0]
    # Set coefficients of each linear term (excluding 'xN') to be zero -> (>= -eps and <= eps)
    constraint_rhs = np.append(constraint_rhs, [eps_value] * 2 * n_linear)
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((2 * n_linear, n_targets * n_features))))

    for i, idx in enumerate(linear_idx_last):
        constraint_lhs[2 * i + last_row, (n_targets - 1) * n_features + idx] = 1
        constraint_lhs[2 * i + 1 + last_row, (n_targets - 1) * n_features + idx] = -1

    ## Coeff of xN to be negative
    # Find the index of the xN term
    xN_idx = feature_names.index(f'x{n_targets - 1}')
    constraint_rhs = np.append(constraint_rhs, -eps_value)
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
    constraint_lhs[-1, (n_targets - 1) * n_features + xN_idx] = 1

    ## Coefficient of xN is in Eq eq and Eq1 are to be negatives of one another 
    constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
    constraint_lhs[-2, (n_targets - 1) * n_features + xN_idx] = 1
    constraint_lhs[-2, xN_idx] = 1
    constraint_lhs[-1, (n_targets - 1) * n_features + xN_idx] = -1
    constraint_lhs[-1, xN_idx] = -1

    last_row = constraint_lhs.shape[0]
    ## All quadratic terms to be zero except for x0 xN and x0 xN-1
    quadratic_idx_last = [i for i, feature_name in enumerate(feature_names) if 
                        ((feature_name.count('x') == 2) and (f'x0 x{n_targets - 1}' not in feature_name 
                        and f'x0 x{n_targets - 2}' not in feature_name)) 
                        or '^2' in feature_name]
    n_quadratic_last = len(quadratic_idx_last)

    # Set coefficients of each quadratic term (excluding 'x0 xN' and 'x0 xN-1') to be zero -> (>= -eps and <= eps)
    constraint_rhs = np.append(constraint_rhs, [eps_value] * 2 * n_quadratic_last)
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((2 * n_quadratic_last, n_targets * n_features))))

    for i, idx in enumerate(quadratic_idx_last):
        constraint_lhs[2 * i + last_row, (n_targets - 1) * n_features + idx] = 1
        constraint_lhs[2 * i + 1 + last_row, (n_targets - 1) * n_features + idx] = -1

    ## Constant term is zero -> (>= -eps and <= eps)
    # Find the index of the constant term '1'
    constant_term_idx = feature_names.index('1')
    constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
    constraint_lhs[-2, (n_targets - 1) * n_features + constant_term_idx] = 1
    constraint_lhs[-1, (n_targets - 1) * n_features + constant_term_idx] = -1

    ## Coeff of x0 xN to be negative
    # Find the index of the x0 xN term
    x0xN_idx = feature_names.index(f'x0 x{n_targets - 1}')
    constraint_rhs = np.append(constraint_rhs, -eps_value)
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
    constraint_lhs[-1, (n_targets - 1) * n_features + x0xN_idx] = 1

    # Coeff of x0 xN and the one in Eq 1 to be equal
    constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
    constraint_lhs[-2, (n_targets - 1) * n_features + x0xN_idx] = 1
    constraint_lhs[-2, x0xN_idx] = -1
    constraint_lhs[-1, (n_targets - 1) * n_features + x0xN_idx] = -1
    constraint_lhs[-1, x0xN_idx] = 1

    ## Coeff of x0 xN-1 to be positive
    # Find the index of the x0 xN-1 term
    x0xN_idx = feature_names.index(f'x0 x{n_targets - 2}')
    constraint_rhs = np.append(constraint_rhs, eps_value)
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((1, n_targets * n_features))))
    constraint_lhs[-1, (n_targets - 1) * n_features + x0xN_idx] = -1

    # Coeff of x0 xN-1 and the one in Eq 1 to be negatives of one another
    constraint_rhs = np.append(constraint_rhs, [eps_value, eps_value])
    constraint_lhs = np.vstack((constraint_lhs, np.zeros((2, n_targets * n_features))))
    constraint_lhs[-2, (n_targets - 1) * n_features + x0xN_idx] = 1
    constraint_lhs[-2, x0xN_idx] = 1
    constraint_lhs[-1, (n_targets - 1) * n_features + x0xN_idx] = -1
    constraint_lhs[-1, x0xN_idx] = -1

    ############################################################
    ### Return constraints
    ############################################################
    return constraint_lhs, constraint_rhs

def extract_and_save_k_values(feature_names, optimizer, output_file='kappa_values.csv'):
    # Calculate k_plus values
    x0sqs_idx = feature_names.index('x0^2')
    x0xN_idx = [i for i, feature_name in enumerate(feature_names) if ('x0' in feature_name) \
            and (feature_name.count('x') == 2)]

    k_plus_idx = [x0sqs_idx] + x0xN_idx
    k_plus_values = [abs(k) for k in optimizer.coef_[0, k_plus_idx]]
    k_plus_values[0] /= 2

    # Calculate k_minus values
    xN_idx = [i for i, feature_name in enumerate(feature_names) if (feature_name.count('x') == 1) \
            and ('x0' not in feature_name) \
            and ('^2' not in feature_name)]
    k_minus_values = [0] + [abs(k) for k in optimizer.coef_[0, xN_idx]]
    k_minus_values[1] /= 2

    # Save k_minus and k_plus values to a file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for k_minus, k_plus in zip(k_minus_values, k_plus_values):
            writer.writerow([k_minus, k_plus])

def load_kappa_values(file_path: Path):

    k_file = file_path / 'kappa_values.csv'

    # Check if the file exists
    if os.path.isfile(k_file):
        with open(k_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            k_values = list(reader)

        k_minus_values = [float(k[0]) for k in k_values]
        k_plus_values = [float(k[1]) for k in k_values]

        return k_minus_values, k_plus_values
    else:
        print(f"Kappa values file ({k_file}) does not exist")
        return None
    

def solve_ODE_system(c0, t_values, km, kp):
    '''
    Solves the ODE system for the given initial conditions and parameters
    params: c0 - initial conditions
            t_values_odes - time values for ODEs
            km_guess_odes - kappa minus values
            kp_guess_odes - kappa plus values
            system_size - size of the system
    returns: N_sol - solution to the ODE system
    '''

    system_size = len(km)
    
    def system_ODEs(clust_conc, t):
            
        # Initialize space for dcdt
        dcdt = list()

        kmc = km * clust_conc
        kpc = kp * clust_conc[0] * clust_conc

        # Treat first and last DEs as special cases
        dcdt.append(np.sum(kmc[2:]) + 2 * kmc[1] - np.sum(kpc[1:]) - 2 * kp[0] * clust_conc[0]**2)
        
        for i in range(1, system_size-1):
            dcdt.append(kpc[i-1] - kpc[i] - kmc[i] + kmc[i+1])

        dcdt.append(-kmc[-1] - kpc[-1] + kp[-2] * clust_conc[0] * clust_conc[-2])
        
        return dcdt
    
    ## Solve ODE system
    N_sol = integrate.odeint(system_ODEs, c0, t_values, rtol=eps_value, atol=eps_value, mxstep=5000)
    
    return N_sol


# This function will solve the ODE system using numbalsoda
def solve_ODE_system_numbalsoda(u0, t_values, km, kp):

    ###################################################################
    def rhs_numbalsoda(t, u, du, p):
        system_size = len(p.km)

        kmc = p.km * u
        kpc = p.kp * u[0] * u

        # Treat first and last DEs as special cases
        du[0] = np.sum(kmc[2:]) + 2 * kmc[1] - np.sum(kpc[1:]) - 2 * p.kp[0] * u[0]**2
        
        for i in range(1, system_size-1):
            du[i] = kpc[i-1] - kpc[i] - kmc[i] + kmc[i+1]

        du[-1] = -kmc[-1] - kpc[-1] + p.kp[-2] * u[0] * u[-2]

    # def rhs_numbalsoda(t, u, du, km, kp):
    #     system_size = len(km)

    #     print('km:', km)
    #     print('kp:', kp)
    #     print('u:', u)
    #     print('du:', du)

    #     kmc = km * u
    #     kpc = kp * u[0] * u

    #     # Treat first and last DEs as special cases
    #     du[0] = np.sum(kmc[2:]) + 2 * kmc[1] - np.sum(kpc[1:]) - 2 * kp[0] * u[0]**2
        
    #     for i in range(1, system_size-1):
    #         du[i] = kpc[i-1] - kpc[i] - kmc[i] + kmc[i+1]

    #     du[-1] = -kmc[-1] - kpc[-1] + kp[-2] * u[0] * u[-2]


    ## use numba.types.Record.make_c_struct to build a c structure.
    ## km_p and kp_p are pointers to the arrays of kappa values
    ## km_len and kp_len are the lengths of the arrays
    args_dtype = types.Record.make_c_struct([
        ('km_p', types.uintp),
        ('km_len', types.int64),
        ('kp_p', types.uintp),
        ('kp_len', types.int64),
    ])

    # This will be a class for our user data
    spec = [
        ('km', types.double[:]),
        ('kp', types.double[:]),
    ]

    @nb.experimental.jitclass(spec)
    class UserData():
        
        def __init__(self):    
            pass
            
        def make_args_dtype(self):
            args = np.zeros(1, dtype=args_dtype)
            args[0][0] = self.km.ctypes.data
            args[0][1] = self.km.shape[0]
            args[0][2] = self.kp.ctypes.data
            args[0][3] = self.kp.shape[0]
            return args
        
        def unpack_pointer(self, user_data_p):
            # Takes in pointer, and unpacks it
            user_data = nb.carray(user_data_p, 1)[0]
            self.km = nb.carray(address_as_void_pointer(user_data.km_p), (user_data.km_len,), dtype=np.float64)
            self.kp = nb.carray(address_as_void_pointer(user_data.kp_p), (user_data.kp_len,), dtype=np.float64)

    # this function will create the numba function to pass to lsoda.
    def create_jit_fcns(rhs, args_dtype, system_size):
        jitted_rhs = nb.njit(rhs)
        @nb.cfunc(types.void(types.double,
                types.CPointer(types.double),
                types.CPointer(types.double),
                types.CPointer(args_dtype)))
        def wrapped_rhs(t, u, du, user_data_p):

            u_array = nb.carray(u, (system_size,), dtype=np.float64)
            du_array = nb.carray(du, (system_size,), dtype=np.float64)
            p = UserData()
            p.unpack_pointer(user_data_p)
            jitted_rhs(t, u_array, du_array, p)

            # user_data = nb.carray(user_data_p, 1)
            # km = nb.carray(address_as_void_pointer(user_data[0].km_p), (user_data[0].km_len,), dtype=np.float64)
            # kp = nb.carray(address_as_void_pointer(user_data[0].kp_p), (user_data[0].kp_len,), dtype=np.float64)
            # jitted_rhs(t, u, du, km, kp)
        
        return wrapped_rhs
    ###################################################################

    system_size = len(km)
    
    p = UserData()
    p.km = np.ascontiguousarray(km)
    p.kp = np.ascontiguousarray(kp)
    args = p.make_args_dtype()

    # km = np.ascontiguousarray(km)
    # kp = np.ascontiguousarray(kp)
    # args = np.array([(km.ctypes.data, km.shape[0], kp.ctypes.data, kp.shape[0])], 
    #                 dtype=args_dtype)

    # Create the function to be called
    rhs_cfunc = create_jit_fcns(rhs_numbalsoda, args_dtype, system_size)

    funcptr = rhs_cfunc.address
    
    # Solve ODE system
    t_eval = np.array(t_values)
    t_span = np.array([min(t_eval), max(t_eval)])
    sol = solve_ivp(funcptr, t_span, u0, 
                    t_eval=t_eval, data=args.ctypes.data, 
                    # rtol=eps_value, atol=eps_value
                    rtol=1e-3, atol=1e-10
                )

    # # u0 = np.ascontiguousarray(u0)
    # sol, success = lsoda(funcptr, u0, t_eval, data = args,
    #                      rtol=eps_value, atol=eps_value)

    return sol.y


if __name__ == '__main__':
    print('Random word:', generate_random_word())



