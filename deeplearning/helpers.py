import numpy as np
import copy
from math import sqrt
from deeplearning.activation_fct import sigmoid, relu, relu_backward, sigmoid_backward, tanh_backward, tanh


def normalize_input(input, mean=None, variance=None):
    """
    Arguments:
    param input -- np.array with data, input.shape = [n, m] where m size of the sample, n nb of features
    param mean -- default=None. If filled, it is used to standardize the input (no mean computation)
    param variance -- default=None. If filled, it is used to standardize the input (no var computation)

    return:

    """
    input_mean, input_var = mean, variance
    if not input_mean:
        input_mean = np.mean(input, axis=1, keepdims=True)
    if not input_var:
        input_var = np.var(input, axis=1, keepdims=True)

    zero_mean_input = input - input_mean
    standardized_input = np.divide(zero_mean_input, input_var)
    return standardized_input, input_mean, input_var


def initialize_parameters_deep(layer_dims, cst_weight_normalization=None):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        if cst_weight_normalization:
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * cst_weight_normalization
        else:
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * sqrt(2./layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z), Z  # trick here, activation cache is Z

    elif activation == "relu":
        A, activation_cache = relu(Z), Z  # trick here, activation cache is Z

    elif activation == "tanh":
        A, activation_cache = tanh(Z), Z  # trick here, activation cache is Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        # A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'tanh')
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


# GRADED FUNCTION: compute_cost

def compute_cost(AL, Y, all_param=None, lambda_reg=0):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = -np.sum(Y * np.log(AL) + (1. - Y) * np.log(1. - AL)) / m

    regularization_cost = 0.
    if lambda_reg:
        assert all_param
        L = len(all_param) // 2
        regularization_cost = sum([np.squeeze(np.sum(np.square(all_param['W' + str(l+1)]))) for l in range(L)])/(2*m)

    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost + lambda_reg * regularization_cost)
    assert (cost.shape == ())
    return cost


def linear_backward(dZ, cache, lambda_reg=0):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    if lambda_reg:
        dW = dW + lambda_reg / m * W
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambda_reg=0):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambda_reg)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, lambda_reg=0):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  "sigmoid", lambda_reg)

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu",
                                                                     lambda_reg)
        # dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "tanh",
        #                                                            lambda_reg)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=10000, lambda_reg=0, print_cost=False,
                  grad_check=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- initial learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    learning_rate_factor = 0

    costs = []  # keep track of cost
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y, parameters, lambda_reg)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, lambda_reg)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            if grad_check:
                gradient_check(parameters, grads, X, Y, 0.000001, lambda_reg)
            costs.append(cost)

        # Update parameters.
        alpha = learning_rate / (1. + i * learning_rate_factor / num_iterations)
        parameters = update_parameters(parameters, grads, alpha)


    # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    return parameters


def gradient_check(parameters, gradients, X, Y, epsilon=1e-7, lambda_reg=0):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    copied_parameters = copy.deepcopy(parameters)

    g1, g2, approx_grads, keys = list(), list(), dict(), list()
    for key, param in copied_parameters.items():
        if key[0] in ('W', 'b'):
            assert 'd' + key in gradients.keys()
            grad_tmp = list()

            for i in range(param.shape[0]):
                for j in range(param.shape[1]):

                    copied_parameters[key][i, j] = parameters[key][i, j] + epsilon
                    AL, _ = L_model_forward(X, copied_parameters)
                    J_p = compute_cost(AL, Y, copied_parameters, lambda_reg)
                    copied_parameters[key][i, j] = parameters[key][i, j] - epsilon
                    AL, _ = L_model_forward(X, copied_parameters)
                    J_m = compute_cost(AL, Y, copied_parameters, lambda_reg)
                    copied_parameters[key][i, j] = parameters[key][i, j]
                    estimated_dJ = (J_p - J_m) / (2.*epsilon)

                    g1.append(gradients['d' + key][i, j])
                    g2.append(estimated_dJ)
                    grad_tmp.append(estimated_dJ)

            keys.append(key)
            approx_grads['d' + key] = np.reshape(grad_tmp, (param.shape[0], param.shape[1]))
            #approx_grads['d' + key] = np.array(grad_tmp)

    g1 = np.array(g1)
    g2 = np.array(g2)

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(g1 - g2)
    denominator = np.linalg.norm(g1) + np.linalg.norm(g2)
    difference = numerator / denominator

    if difference > 2e-6:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        for key in approx_grads.keys():
            numerator2 = np.linalg.norm(gradients[key] - approx_grads[key])
            denominator2 = np.linalg.norm(gradients[key]) + np.linalg.norm(approx_grads[key])
            difference2 = numerator2 / denominator2
            print(key, '-->', difference2)
            if difference2 > 2e-6:
                print('--- computed_grad ---')
                print(gradients[key])
                print('--- approx grad ---')
                print(approx_grads[key])

    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference



