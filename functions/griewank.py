import numpy as np
from _function import Function


class Griewank(Function):
    r"""Griewank function.
        f(\bm{x}) = 1 + \frac{1}{4000} \sum_{i=1}^{n} x_i^2 - \prod_{i=1}^{n} \cos\left(\frac{x_i}{\sqrt{i}}\right), \bm{x} \in \mathbb{R}^n.

    Global minimum at x_{min} = [0, 0, ..., 0], f(x_{min}) = 0.    
    """

    @property
    def global_minimum(self) -> np.ndarray:
        return np.array([0.0, 0.0])
    
    @property
    def search_domain(self) -> tuple[float, float]:
        return (-100.0, 100.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Griewank function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Function value(s) at the input point(s).
        """

        if x.ndim == 1:
            x = x[np.newaxis, :]
        idx = np.arange(1, x.shape[1] + 1).astype(np.float32)
        return np.sum(x**2, axis=-1)/4000 - np.prod(np.cos(x / np.sqrt(idx)), axis=-1) + 1


if __name__ == "__main__":
    f = Griewank()
    x = np.random.uniform(f.search_domain[0], f.search_domain[1], (5, 3))
    y = f(x)
    for i in range(x.shape[0]):
        print(f"Griewank function value at {x[i]}: {y[i]}")
        assert y[i] == f(x[i])

    print(f"Griewank function value at global minimum {f.global_minimum}: {f(f.global_minimum)}")
    assert np.isclose(f(f.global_minimum), 0.0)
