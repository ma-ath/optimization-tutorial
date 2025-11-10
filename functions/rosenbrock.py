import numpy as np
from _function import Function


class Rosenbrock(Function):
    r"""Rosenbrock function.
        f(\bm{x}) = \sum_{i=1}^{n-1} \left(100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right), \bm{x} \in \mathbb{R}^n.

    Global minimum at x_{opt} = [1, 1, ..., 1], f(x_{opt}) = 0.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Sphere function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Function value(s) at the input point(s).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return np.sum(100*(x[:, 1:] - x[:, :-1]**2)**2 + (x[:, :-1] - 1)**2, axis=-1)


if __name__ == "__main__":
    f = Rosenbrock()
    x = np.array([[1.0, 1.0, 1.0], 
                  [0.0, 0.0, 0.0],
                  [-1.0, 1.0, 1.0]])
    y = f(x)
    for i in range(x.shape[0]):
        assert y[i] == f(x[i])
