import random
from nltk.corpus import words
import numpy as np
import os
import csv

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


if __name__ == '__main__':
    print('Random word:', generate_random_word())



