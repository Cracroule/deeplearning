import numpy as np
from tests.utils import assert_content_equality
from deeplearning.activation_fct import relu_backward, sigmoid_backward, d_relu, d_sigmoid


def test_d_relu():
    z_l = [[0.45, 0.48, 0.87, -0.2, 1.4, -4.2]]
    z = np.array([np.array(xi) for xi in z_l])
    z_prime = d_relu(z)
    assert_content_equality(z_prime, [[1, 1, 1, 0, 1, 0]])


def test_relu_backward():
    dA_l = [[-0.41675785, -0.05626683]]
    z_l = [[0.04153939, -1.11792545]]
    dA = np.array([np.array(xi) for xi in dA_l])
    Z = np.array([np.array(xi) for xi in z_l])
    dZ = relu_backward(dA, Z)
    dZ_res = [[-0.41675785, 0.]]
    assert_content_equality(dZ, dZ_res)


def test_sigmoid_backward():
    dA_l = [[-0.41675785, -0.05626683]]
    z_l = [[0.04153939, -1.11792545]]
    dA = np.array([np.array(xi) for xi in dA_l])
    Z = np.array([np.array(xi) for xi in z_l])
    dZ = sigmoid_backward(dA, Z)
    dZ_res = [[-0.10414453, -0.01044791]]
    assert_content_equality(dZ, dZ_res)