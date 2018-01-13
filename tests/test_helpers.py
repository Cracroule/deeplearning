import numpy as np
from tests.utils import assert_content_equality
from deeplearning.helpers import initialize_parameters_deep, linear_forward, \
    linear_activation_forward, compute_cost, linear_backward, linear_activation_backward, L_model_forward, \
    L_model_backward, update_parameters


def test_initialize_parameters_deep():
    parameters = initialize_parameters_deep([5, 4, 3], cst_weight_normalization=0.01)
    W1 = [[0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
          [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
          [-0.01313865, 0.00884622, 0.00881318, 0.01709573, 0.00050034],
          [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]]
    b1 = [[0.], [0.], [0.], [0.]]
    W2 = [[-0.01185047, -0.0020565, 0.01486148, 0.00236716],
          [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
          [-0.00768836, -0.00230031, 0.00745056, 0.01976111]]
    b2 = [[0.], [0.], [0.]]
    assert_content_equality(parameters["W1"], W1)
    assert_content_equality(parameters["W1"], W1)
    assert_content_equality(parameters["b1"], b1)
    assert_content_equality(parameters["W2"], W2)
    assert_content_equality(parameters["b2"], b2)


def test_linear_forward_test_case():
    a_l = [[1.62434536, -0.61175641], [-0.52817175, -1.07296862], [0.86540763, -2.3015387]]
    w_l = [[1.74481176, -0.7612069, 0.3190391]]
    b_l = [[-0.24937038]]
    A = np.array([np.array(xi) for xi in a_l])
    W = np.array([np.array(xi) for xi in w_l])
    b = np.array([np.array(xi) for xi in b_l])
    Z, linear_cache = linear_forward(A, W, b)
    assert_content_equality(Z, [[3.26295337, -1.23429987]])


def test_linear_activation_forward_test_case():
    a_l = [[-0.41675785, -0.05626683], [-2.1361961, 1.64027081], [-1.79343559, -0.84174737]]
    w_l = [[0.50288142, -1.24528809, -1.05795222]]
    b_l = [[-0.90900761]]
    A_prev = np.array([np.array(xi) for xi in a_l])
    W = np.array([np.array(xi) for xi in w_l])
    b = np.array([np.array(xi) for xi in b_l])
    A_sigm, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
    A_relu, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
    assert_content_equality(A_sigm, [[ 0.96890023, 0.11013289]])
    assert_content_equality(A_relu, [[ 3.43896131, 0.]])


def test_L_model_forward():
    # a bit complicated to smoke test
    # TODO later on
    pass


def test_compute_cost():
    a_l = [[0.8, 0.9, 0.4]]
    y_l = [[1, 1, 1]]
    AL = np.array([np.array(xi) for xi in a_l])
    Y = np.array([np.array(xi) for xi in y_l])
    cost = compute_cost(AL, Y)
    assert assert_content_equality(cost, 0.414931599615)


def test_linear_backward():
    dz_l = [[1.62434536, -0.61175641]]
    a_prev_l = [[-0.52817175, -1.07296862], [0.86540763, -2.3015387], [1.74481176, -0.7612069]]
    w_l = [[0.3190391, -0.24937038, 1.46210794]]
    b_l = [[-2.06014071]]
    dZ = np.array([np.array(xi) for xi in dz_l])
    A_prev = np.array([np.array(xi) for xi in a_prev_l])
    W = np.array([np.array(xi) for xi in w_l])
    b = np.array([np.array(xi) for xi in b_l])
    dA_prev, dW, db = linear_backward(dZ, (A_prev, W, b))
    dA_prev_res = [[0.51822968, -0.19517421], [-0.40506361, 0.15255393], [2.37496825, -0.89445391]]
    dW_res = [[-0.10076895,  1.40685096,  1.64992505]]
    db_res = [[0.50629448]]
    assert_content_equality(dA_prev, dA_prev_res)
    assert_content_equality(dW, dW_res)
    assert_content_equality(db, db_res)


def test_linear_activation_backward():
    A_prev = np.array([np.array(xi) for xi in [[-2.1361961, 1.64027081], [-1.79343559, -0.84174737],
                                               [0.50288142, -1.24528809]]])
    W = np.array([np.array(xi) for xi in [[-1.05795222, -0.90900761, 0.55145404]]])
    b = np.array([np.array(xi) for xi in [[2.29220801]]])
    linear_cache = (A_prev, W, b)
    dA_l = [[-0.41675785, -0.05626683]]
    z_l = [[0.04153939, -1.11792545]]
    dA = np.array([np.array(xi) for xi in dA_l])
    Z = np.array([np.array(xi) for xi in z_l])
    cache = linear_cache, Z
    dA_prev_relu, dW_relu, db_relu = linear_activation_backward(dA, cache, "relu")
    dA_prev_sigm, dW_sigm, db_sigm = linear_activation_backward(dA, cache, "sigmoid")
    dA_prev_sigm_res = [[0.11017994, 0.01105339], [0.09466817, 0.00949723], [-0.05743092, -0.00576154]]
    dW_sigm_res = [[0.10266786, 0.09778551, -0.01968084]]
    db_sigm_res = [[-0.05729622]]
    dA_prev_relu_res = [[0.44090989, 0.], [0.37883606, 0.], [-0.2298228, 0.]]
    dW_relu_res = [[0.44513824, 0.37371418, -0.10478989]]
    db_relu_res = [[-0.20837892]]
    assert_content_equality(dA_prev_sigm, dA_prev_sigm_res)
    assert_content_equality(dW_sigm, dW_sigm_res)
    assert_content_equality(db_sigm, db_sigm_res)
    assert_content_equality(dA_prev_relu, dA_prev_relu_res)
    assert_content_equality(dW_relu, dW_relu_res)
    assert_content_equality(db_relu, db_relu_res)


def test_L_model_backward():
    #TODO
    pass


def test_update_parameters():
    #TODO
    pass
