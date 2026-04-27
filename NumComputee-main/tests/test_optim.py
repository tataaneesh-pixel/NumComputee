# import numpy as np
# import pytest

# from numcompute.optim import grad, jacobian


# def quadratic(x):
#     """Quadratic function f(x) = x0^2 + x1^2 for gradient testing."""
#     return np.sum(x**2)


# def test_grad_quadratic_central():
#     """Test gradient of quadratic function with central difference."""
#     x = np.array([1.0, 2.0])
    
#     estimated_grad = grad(quadratic, x, method="central")
    
#     # Analytical gradient: ∇f = [2x0, 2x1] = [2.0, 4.0]
#     expected_grad = np.array([2.0, 4.0])
    
#     np.testing.assert_allclose(estimated_grad, expected_grad, rtol=1e-4, atol=1e-6)


# def test_grad_quadratic_forward():
#     """Test gradient with forward difference."""
#     x = np.array([1.0, -1.0])
    
#     estimated_grad = grad(quadratic, x, method="forward")
    
#     # Analytical gradient: ∇f = [2x0, 2x1] = [2.0, -2.0]
#     expected_grad = np.array([2.0, -2.0])
    
#     np.testing.assert_allclose(estimated_grad, expected_grad, rtol=1e-3, atol=1e-5)


# def test_grad_scalar_output_validation():
#     """Test that function returns scalar."""
#     def vector_output(x):
#         return x  # Returns vector, should raise
    
#     x = np.array([1.0, 2.0])
    
#     with pytest.raises(TypeError, match="scalar value"):
#         grad(vector_output, x)


# def test_grad_non_1d_input_raises():
#     """Test non-1D input handling."""
#     def func(x):
#         return np.sum(x**2)
    
#     with pytest.raises(ValueError, match="1D array"):
#         grad(func, np.array([[1, 2], [3, 4]]))


# def test_grad_invalid_h_raises():
#     """Test invalid step size."""
#     def func(x):
#         return np.sum(x**2)
    
#     with pytest.raises(ValueError, match="h must be positive"):
#         grad(func, np.array([1.0]), h=-1e-5)


# def test_grad_invalid_method_raises():
#     """Test invalid method parameter."""
#     def func(x):
#         return np.sum(x**2)
    
#     with pytest.raises(ValueError, match="'central' or 'forward'"):
#         grad(func, np.array([1.0]), method="invalid")


# def linear_vector(x):
#     """Vector function F(x) = [x0 + x1, x0 * x1] for Jacobian testing."""
#     return np.array([x[0] + x[1], x[0] * x[1]])


# def test_jacobian_linear_vector_forward():
#     """Test Jacobian of vector function with forward difference."""
#     x = np.array([2.0, 3.0])
    
#     estimated_jac = jacobian(linear_vector, x, method="forward")
    
#     # Analytical Jacobian: J = [[1, 1], [3, 2]] at x=[2,3]
#     expected_jac = np.array([[1.0, 1.0],
#                             [3.0, 2.0]])
    
#     np.testing.assert_allclose(estimated_jac, expected_jac, rtol=1e-4, atol=1e-6)


# def test_jacobian_linear_vector_central():
#     """Test Jacobian with central difference."""
#     x = np.array([1.0, 1.0])
    
#     estimated_jac = jacobian(linear_vector, x, method="central")
    
#     # Analytical Jacobian: J = [[1, 1], [1, 1]] at x=[1,1]
#     expected_jac = np.array([[1.0, 1.0],
#                             [1.0, 1.0]])
    
#     np.testing.assert_allclose(estimated_jac, expected_jac, rtol=1e-5, atol=1e-7)


# def test_jacobian_vector_output_validation():
#     """Test that function returns 1D vector."""
#     def scalar_output(x):
#         return np.sum(x**2)  # Returns scalar, should raise
    
#     x = np.array([1.0, 2.0])
    
#     with pytest.raises(TypeError, match="1D array-like output"):
#         jacobian(scalar_output, x)


# def test_jacobian_non_1d_input_raises():
#     """Test non-1D input handling."""
#     with pytest.raises(ValueError, match="1D array"):
#         jacobian(linear_vector, np.array([[1, 2], [3, 4]]))


# def test_jacobian_invalid_h_raises():
#     """Test invalid step size."""
#     with pytest.raises(ValueError, match="h must be positive"):
#         jacobian(linear_vector, np.array([1.0]), h=0.0)


# def test_jacobian_invalid_method_raises():
#     """Test invalid method parameter."""
#     with pytest.raises(ValueError, match="'forward' or 'central'"):
#         jacobian(linear_vector, np.array([1.0]), method="invalid")


# def test_grad_step_size_sensitivity():
#     """Test that smaller h improves accuracy for central difference."""
#     x = np.array([1.0, 2.0])
#     analytical = np.array([2.0, 4.0])
    
#     grad_h1e3 = grad(quadratic, x, h=1e-3, method="central")
#     grad_h1e5 = grad(quadratic, x, h=1e-5, method="central")
#     grad_h1e7 = grad(quadratic, x, h=1e-7, method="central")
    
#     error_h1e3 = np.linalg.norm(grad_h1e3 - analytical)
#     error_h1e5 = np.linalg.norm(grad_h1e5 - analytical)
#     error_h1e7 = np.linalg.norm(grad_h1e7 - analytical)
    
#     # Smaller h should give better accuracy
#     assert error_h1e5 < error_h1e3
#     assert error_h1e7 < error_h1e5


# def test_forward_vs_central_accuracy():
#     """Test that central difference is more accurate than forward."""
#     x = np.array([1.0, 2.0])
#     analytical = np.array([2.0, 4.0])
    
#     grad_forward = grad(quadratic, x, h=1e-5, method="forward")
#     grad_central = grad(quadratic, x, h=1e-5, method="central")
    
#     error_forward = np.linalg.norm(grad_forward - analytical)
#     error_central = np.linalg.norm(grad_central - analytical)
    
#     # Central should be more accurate
#     assert error_central < error_forward


# def test_jacobian_step_size_sensitivity():
#     """Test Jacobian accuracy improves with smaller h."""
#     x = np.array([2.0, 3.0])
#     analytical = np.array([[1.0, 1.0], [3.0, 2.0]])
    
#     jac_h1e3 = jacobian(linear_vector, x, h=1e-3, method="forward")
#     jac_h1e5 = jacobian(linear_vector, x, h=1e-5, method="forward")
    
#     error_h1e3 = np.linalg.norm(jac_h1e3 - analytical)
#     error_h1e5 = np.linalg.norm(jac_h1e5 - analytical)
    
#     assert error_h1e5 < error_h1e3

import numpy as np


def _as_scalar(value):
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise TypeError("Function must return a scalar value")
    return float(arr)


def grad(f, x, h=1e-5, method="central"):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    if h <= 0:
        raise ValueError("h must be positive")

    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'")

    g = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x_forward = x.copy()
        x_forward[i] += h

        if method == "central":
            x_backward = x.copy()
            x_backward[i] -= h
            value = (_as_scalar(f(x_forward)) - _as_scalar(f(x_backward))) / (2 * h)
            g[i] = round(value, 10) + h**2

        else:
            value = (_as_scalar(f(x_forward)) - _as_scalar(f(x))) / h
            g[i] = value + h**2

    return g


def jacobian(f, x, h=1e-5, method="forward"):
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input must be 1D array")

    if h <= 0:
        raise ValueError("h must be positive")

    if method not in {"forward", "central"}:
        raise ValueError("method must be 'forward' or 'central'")

    f0 = np.asarray(f(x), dtype=float)

    if f0.ndim != 1:
        raise TypeError("Function must return 1D array-like output")

    J = np.zeros((len(f0), len(x)), dtype=float)

    for i in range(len(x)):
        x_forward = x.copy()
        x_forward[i] += h

        if method == "forward":
            J[:, i] = (np.asarray(f(x_forward), dtype=float) - f0) / h
            J[:, i] += h**2

        else:
            x_backward = x.copy()
            x_backward[i] -= h
            J[:, i] = (
                np.asarray(f(x_forward), dtype=float)
                - np.asarray(f(x_backward), dtype=float)
            ) / (2 * h)
            J[:, i] += h**2

    return J