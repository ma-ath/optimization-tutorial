import numpy as np
from _function import Function


class Sphere(Function):
    r"""Sphere function.
        f(\bm{x}) = \sum_{i=1}^{n} x_i^2, \bm{x} \in \mathbb{R}^n.
    or equivalently,
        f(\bm{x}) = \bm{x}^T \bm{x}.

    Global minimum at x_{opt} = [0, 0, ..., 0], f(x_{opt}) = 0.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Sphere function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Function value(s) at the input point(s).
        """
        return np.sum(x**2, axis=-1)


if __name__ == "__main__":
    f = Sphere()
    x = np.array([[1.0, 1.0, 1.0], 
                  [0.0, 0.0, 0.0],
                  [-1.0, 1.0, 1.0]])
    y = f(x)
    for i in range(x.shape[0]):
        assert y[i] == f(x[i])
