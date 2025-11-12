import numpy as np

from src.functions._function import Function


class Rosenbrock(Function):
    r"""Rosenbrock function.
        f(\bm{x}) = \sum_{i=1}^{n-1} \left(100 (x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right), \bm{x} \in \mathbb{R}^n.

    Global minimum at x_{min} = [1, 1, ..., 1], f(x_{min}) = 0.
    """

    @property
    def global_minimum(self) -> np.ndarray:
        return np.array([1.0, 1.0])
    
    @property
    def function_domain(self) -> tuple[float, float]:
        return (-2.0, 2.0)

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
    x = np.random.uniform(f.function_domain[0], f.function_domain[1], (5, 3))
    y = f(x)
    for i in range(x.shape[0]):
        print(f"Rosenbrock function value at {x[i]}: {y[i]}")
        assert y[i] == f(x[i])

    print(f"Rosenbrock function value at global minimum {f.global_minimum}: {f(f.global_minimum)}")
    assert np.isclose(f(f.global_minimum), 0.0)
