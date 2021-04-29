from numpy import linalg as lg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def __find_inverse_of_sigma(sigma, threshold):
    """
    Finds the Moore-Penrose inverse of sigma.
    :param sigma: A 1-d (i.e., diagonal) array
    :param threshold: Beneath this value, the singular values are considered zero.
    :return: The array in which we invert every nonzero singular value, while the rest remain zero
    """
    is_negative = np.absolute(sigma) < threshold  # A logical array, 1's where the SVs are zero
    sigma_no_zeros = sigma + is_negative  # Before taking inverse, we remove zeros
    sigma_reversed = (sigma_no_zeros ** (-1)) * (1 - is_negative)  # at the end, vanish the zero SVs
    return sigma_reversed


def fit_linear_regression(design_matrix, y):
    """
    Finds the linear regression fit for the design matrix X and response vector y
    :return: a pair (w, sigma_mat), where w is the coefficient vector, and S is the array of singular values of X.
    """
    sigma_mat = lg.svd(design_matrix, compute_uv=False)
    return lg.pinv(design_matrix) @ y, sigma_mat


def predict(design_matrix, w_vector):
    """
    Returns a numpy array with the predicted values of the model.
    """
    return design_matrix @ w_vector


def mse(response_vector, prediction_vector):
    """
    Returns the mean square error.
    """
    return ((response_vector - prediction_vector) ** 2).mean()


def load_data(path):
    """
    Load and process data from csv file into array.
    :return A pair (X, y) where X is the design matrix and y is the response vector
    """
    data = pd.read_csv(path, sep=',', header=None).values
    response_y = data[1:, 2].astype(float)
    response_y = np.where(np.isfinite(response_y), response_y, 0)

    data = np.delete(data, [0, 1, 2, 14, 16, 17, 18], 1)
    titles = data[0, :]
    data = np.delete(data, 0, 0)
    design_matrix = data.astype(float)
    design_matrix = np.where(np.isfinite(design_matrix), design_matrix, 0)
    return design_matrix, response_y, titles


def plot_singular_values(singular_values):
    """
    Plots singular values.
    """
    x_values = np.arange(1, singular_values.size + 1)
    plt.plot(x_values, singular_values, "-or")
    plt.title("Scree-plot")
    plt.xlabel("# of singular value")
    plt.ylabel("Singular values")
    plt.show()


def divide_into_training(design_matrix, y):
    """
    Chooses randomly 3/4 of the data for training, and the rest is for testing.
    """
    n_rows, n_cols = design_matrix.shape
    binary_random_quarter = (np.random.randint(0, 4, n_rows) == 0)
    design_matrix_training, y_training = design_matrix[~binary_random_quarter], y[~binary_random_quarter]
    design_matrix_test, y_test = design_matrix[binary_random_quarter], y[binary_random_quarter]
    return design_matrix_training, y_training, design_matrix_test, y_test


def fit_model_p_percent(training_X, training_y, test_X, test_y, p):
    """
    Runs training on a portion p (0 < p < 1) of the training data.
    """
    n_rows, n_cols = training_X.shape
    max_row = int(n_rows * p)
    w_p, _ = fit_linear_regression(training_X[:max_row + 1, :], training_y[:max_row + 1])
    return mse(test_X @ w_p, test_y)


def run_iterative_prediction(path):
    """
    Runs training for p = 1,...,100 percent of the training data, and compares to the test
    :param path: Path to csv
    """
    entire_X, entire_y, _ = load_data(path)
    training_X, training_y, test_X, test_y = divide_into_training(entire_X, entire_y)
    mse_matrix = [fit_model_p_percent(training_X, training_y, test_X, test_y, p/100)
                  for p in range(1, 101)]
    x_values = np.arange(1, 101)
    plt.plot(x_values, mse_matrix, "-ob")
    plt.title("MSE as function of % of training data used")
    plt.xlabel("Percent of raining data used for the model")
    plt.ylabel("Mean square error")
    plt.show()


def calculate_peterson(vector1, vector2):
    return np.cov(vector1, vector2)[0][1] / (np.std(vector1) * np.std(vector2))


def feature_evaluation(design_matrix, response_vector):
    """
    Plots, for every non-categorical feature, feature values and response values.
    (We receive the design matrix already without categorical data.)
    """
    # I parse the titles earlier but I don't want to change the given function API:
    TITLES = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
              'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_renovated',
              'sqft_living15', 'sqft_lot15']
    n_rows, n_cols = design_matrix.shape
    for i in range(n_cols):
        plt.scatter(design_matrix[:, i], response_vector, c='g')

        plt.title("Prices (response) as function of " + TITLES[i] + '\n'
                   + "Pearson Correlation is " + str(calculate_peterson(design_matrix[:, i], response_vector)))

        plt.ylabel("House price")
        plt.xlabel(TITLES[i])
        plt.show()
        plt.savefig('Prices as function of ' + TITLES[i])


if __name__ == "__main__":
    run_iterative_prediction('kc_house_data.csv')
    X, y, titles = load_data('kc_house_data.csv')
    w, S = fit_linear_regression(X, y)
    np.savetxt('w.csv', w, delimiter=',')
    np.savetxt('Singular_values.csv', S, delimiter=',')
    plot_singular_values(S)
    feature_evaluation(X, y)
    # tar -cvf ex_2_Roy_Shtoyerman.tar Ex2.pdf linear_model.py
