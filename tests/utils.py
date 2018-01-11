import numpy as np


# homemade comparison tool
def assert_content_equality(a, b, eps=10e-7):
    if isinstance(a, tuple) or isinstance(a, list) or isinstance(a, np.ndarray):
        assert isinstance(b, tuple) or isinstance(b, list) or isinstance(b, np.ndarray)
        assert len(a) == len(b)
        for i, e in enumerate(a):
            assert assert_content_equality(e, b[i], eps)
    else:
        assert abs(a - b) <= eps
    return True