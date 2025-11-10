import numpy as np
from _function import Function


class Rastrigin(Function):
    r"""Rastrigin function.
        f(\bm{x}) = A n + \sum_{i=1}^{n} \left(x_i^2 - A \cos(2 \pi x_i)\right), \bm{x} \in \mathbb{R}^n.

    Global minimum at x_{min} = [0, 0, ..., 0], f(x_{min}) = 0.
    """
    def __init__(self, A: float = 10.0):
        self._A = A
        super().__init__()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Rastrigin function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Function value(s) at the input point(s).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return self._A*x.shape[1] + np.sum(x**2 - self._A*np.cos(2*np.pi*x), axis=1)


if __name__ == "__main__":
    f = Rastrigin()
    x = np.array([[1.0, 1.0, 1.0], 
                  [0.0, 0.0, 0.0],
                  [-1.0, 1.0, 1.0]])
    y = f(x)
    print(f"Rastrigin function value at {x}: {y}")
    for i in range(x.shape[0]):
        assert y[i] == f(x[i])
