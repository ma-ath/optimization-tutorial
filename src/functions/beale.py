import numpy as np

from src.functions._function import Function


class Beale(Function):
    r"""Beale function.
    The Beale function is a well-known test problem for optimization algorithms.
    It is defined as:
    .. math::
        f(x_1, x_2) = (1.5 - x_1 + x_1 x_2)^2 + (2.25 - x_1 + x_1 x_2^2)^2 + (2.625 - x_1 + x_1 x_2^3)^2
    where :math:`x = [x_1, x_2]`.
    It has a global minimum at:
    .. math::
        x_{min} = [3, 0.5], f(x_{min}) = 0
    Attributes:
        Global minimum at x_{min} = [3, 0.5], f(x_{min}) = 0.
        Search domain: x_i ∈ [-4.5, 4.5] for i = 1, 2.
        Differentiable: True
    """

    @property
    def global_minimum(self) -> np.ndarray:
        return np.array([3.0, 0.5])
    
    @property
    def function_domain(self) -> tuple[float, float]:
        return (-4.5, 4.5)

    @property
    def is_differentiable(self) -> bool:
        return True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Beale function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Function value(s) at the input point(s).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[-1] != 2:
            raise ValueError("Beale function is only defined for 2-dimensional input.")
        x1, x2 = x[:, 0], x[:, 1]
        return  ((1.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2).squeeze()


    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the Beale function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Gradient vector(s) at the input point(s).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[-1] != 2:
            raise ValueError("Beale function is only defined for 2-dimensional input.")
        x1, x2 = x[:, 0], x[:, 1]
        f1 = 1.5 - x1 + x1 * x2
        f2 = 2.25 - x1 + x1 * x2**2
        f3 = 2.625 - x1 + x1 * x2**3
        df_dx1 = 2 * f1 * (-1 + x2) + 2 * f2 * (-1 + x2**2) + 2 * f3 * (-1 + x2**3)
        df_dx2 = 2 * f1 * (x1) + 4 * f2 * (x1 * x2) + 6 * f3 * (x1 * x2**2)
        return np.stack([df_dx1, df_dx2], axis=-1).squeeze()


if __name__ == "__main__":
    f = Beale()
    x = np.random.uniform(f.function_domain[0], f.function_domain[1], (5, 2))
    x_grad = f.gradient(x)
    y = f(x)
    for i in range(x.shape[0]):
        print(f"Beale function value at {x[i]}: {y[i]}")
        assert y[i] == f(x[i])
        assert (x_grad[i] == f.gradient(x[i])).all()

    print(f"Beale function value at global minimum {f.global_minimum}: {f(f.global_minimum)}")
    assert np.isclose(f(f.global_minimum), 0.0)

    x_grad = f.gradient(f.global_minimum)
    print(f"Gradient at global minimum {f.global_minimum    }:{x_grad}")
    assert np.allclose(x_grad, 0.0, atol=1e-4)
