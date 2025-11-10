import numpy as np
from _function import Function


class Levy(Function):
    r"""Levy function.
        f(\bm{x}) = \sin^2(\pi x_1) + \sum_{i=1}^{n-1} (x_i - 1)^2 (1 + 10 \sin^2(\pi x_i + 1)) + (x_n - 1)^2 (1 + \sin^2(2\pi x_n)), \bm{x} \in \mathbb{R}^n.

    Global minimum at x_{min} = [1, 1, ..., 1], f(x_{min}) = 0.
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
        w = 1 + (x - 1)/4
        term1 = np.sin(np.pi*w[:, 0])**2
        term3 = (w[:, -1]-1)**2 * (1 + np.sin(2*np.pi*w[:, -1])**2)
        term2 = np.sum((w[:, :-1]-1)**2 * (1 + 10*np.sin(np.pi*w[:, :-1]+1)**2), axis=1)
        return term1 + term2 + term3

if __name__ == "__main__":
    f = Levy()
    x = np.array([[1.0, 1.0, 1.0], 
                  [0.0, 0.0, 0.0],
                  [-1.0, 1.0, 1.0]])
    y = f(x)
    print(f"Ackley function value at {x}: {y}")
    for i in range(x.shape[0]):
        assert y[i] == f(x[i])
