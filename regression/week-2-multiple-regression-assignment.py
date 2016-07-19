"""
Regression Week 2: Multiple Regression (gradient descent)
"""
"""
About Numpy.
Matrix operations are a significant part of the inner loop of the algorithms
we will implement in this module and subsequent modules.
To speed up your code, it can be important to use a specialized matrix operations library.
There are many such libraries out there.
In Python, we recommend Numpy, a popular open-source package for this task.
"""
import graphlab
import numpy as np

sales = graphlab.SFrame('data/kc_house_data.gl/')

def get_numpy_data(data_sframe, features, output):
    """ transfer gl data into numpy arrays
    input:
        data_sframe: gl data sframe
        features: a list of feature column names
        output: target column name
    output:
        feature_matrix: 2-Dimension N by D+1 numpy array (N examples, D+1 features)
        output_array: N by 1 numpy array contains all targets of N examples
    """
    data_sframe['constant'] = 1 #add a constant column to an SFrame
    features = ['constant'] + features
    # sframe[a list of column names] -> SFrame
    # sframe[colum name] -> SArray
    # SFrame.to_numpy(), SArray.to_numpy()
    feature_sframe = data_sframe[features]
    feature_matrix = feature_sframe.to_numpy()
    output_sarray = data_sframe[output]
    output_array = output_sarray.to_numpy()

    return(feature_matrix, output_array)

"""
np.dot() also works when dealing with a matrix and a vector.
(dimension should be matched)
Recall that the predictions from all the observations
is just the RIGHT (as in weights on the right) dot product
between the features matrix and the weights vector.
"""
def predict_output(feature_matrix, weights):
    """  to compute the predictions for an entire matrix of features given the matrix and the weights """
    # assume feature_matrix is a numpy matrix containing the
    #   features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions=np.dot(feature_matrix, weights)
    return(predictions)

""" Computing the derivative
    We are now going to move to computing the derivative of the regression cost function.
    Recall that the cost function is the sum over the data points of the squared difference
    between an observed output and a predicted output.

    Since the derivative of a sum is the sum of the derivatives we can compute the derivative
    for a single data point and then sum over data points.
    We can write the squared difference between the observed output and predicted output
    for a single point as follows:

(w[0]*[CONSTANT] + w[1]*[feature_1] + ... + w[i] *[feature_i] + ... + w[k]*[feature_k] - output)^2
Where we have k features and a constant. So the derivative with respect to weight w[i] by the chain rule is:
2*(w[0]*[CONSTANT] + w[1]*[feature_1] + ... + w[i] *[feature_i] + ... + w[k]*[feature_k] - output)* [feature_i]
The term inside the paranethesis is just the error (difference between prediction and output). So we can re-write this as:
        2*error*[feature_i]
That is, the derivative for the weight for feature i is the sum (over data points) of 2 times
the product of the error and the feature itself.
In the case of the constant then this is just twice the sum of the errors!
Recall that twice the sum of the product of two vectors is just twice the dot product of the two vectors.
Therefore the derivative for the weight for feature_i is just two times the dot product
between the values of feature_i and the current errors.

"""
def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative=2*np.dot(errors, feature)
    return(derivative)

""" Gradient descent
Now we will write a function that performs a gradient descent.
The basic premise is simple.
Given a starting point we update the current weights by moving in the negative gradient direction.
Recall that the gradient is the direction of increase
and therefore the negative gradient is the direction of decrease and we're trying to minimize a cost function.
The amount by which we move in the negative gradient direction is called the 'step size'.
We stop when we are 'sufficiently close' to the optimum.
We define this by requiring that the magnitude (length) of the gradient vector to be smaller than a fixed 'tolerance'.

A few things to note before we run the gradient descent.
Since the gradient is a sum over all the data points
and involves a product of an error
and a feature the gradient itself will be very large
since the features are large (squarefeet) and the output is large (prices).
So while you might expect "tolerance" to be small, small is only relative to the size of the features.

For similar reasons the step size will be much smaller than
you might expect but this is because the gradient has such large values.
"""
from math import sqrt
# recall that the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)
def regression_gradient_descent_v1(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights) # make sure it's a numpy array
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = 2*np.dot(errors,feature_matrix[:,i]) ## for all the data points
            # add the squared value of the derivative to the gradient magnitude (for assessing convergence)
            gradient_sum_squares += derivative**2
            # subtract the step size times the derivative from the current weight
            weights[i] -= step_size*derivative
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)

def regression_gradient_descent_v2(feature_matrix, output, initial_weights, step_size, tolerance):
    gradient_magnitude = tolerance + 1
    weights = np.array(initial_weights)

    while gradient_magnitude > tolerance:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions - output
        derivative = feature_derivative(errors, feature_matrix)
        weights -= step_size*derivative
        gradient_magnitude = sqrt(np.sum(derivative**2))

    return(weights)


## Running

train_data,test_data = sales.random_split(.8,seed=0)
# let's test out the gradient descent
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights=regression_gradient_descent_v1(simple_feature_matrix,output,initial_weights,step_size,tolerance)
print simple_weights

simple_weights=regression_gradient_descent_v2(simple_feature_matrix,output,initial_weights,step_size,tolerance)
print simple_weights

model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors.
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

multiple_weights=regression_gradient_descent_v1(feature_matrix,output,initial_weights,step_size,tolerance)
print multiple_weights

multiple_weights=regression_gradient_descent_v2(feature_matrix,output,initial_weights,step_size,tolerance)
print multiple_weights
