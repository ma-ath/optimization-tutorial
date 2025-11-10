import numpy as np

from _function import Function


class Sphere(Function):
    r"""Sphere function.
        f(\bm{x}) = \sum_{i=1}^{n} x_i^2, \bm{x} \in \mathbb{R}^n.
    or equivalently,
        f(\bm{x}) = \bm{x}^T \bm{x}.

    Global minimum at x_{min} = [0, 0, ..., 0], f(x_{min}) = 0.
    """

    @property
    def global_minimum(self) -> np.ndarray:
        return np.array([0.0, 0.0])
    
    @property
    def search_domain(self) -> tuple[float, float]:
        return (-100.0, 100.0)

    @property
    def is_differentiable(self) -> bool:
        return True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Sphere function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Function value(s) at the input point(s).
        """
        return np.sum(x**2, axis=-1)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the Sphere function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Gradient vector(s) at the input point(s).
        """
        return 2 * x


if __name__ == "__main__":
    f = Sphere()
    x = np.random.uniform(f.search_domain[0], f.search_domain[1], (5, 3))
    y = f(x)
    for i in range(x.shape[0]):
        print(f"Sphere function value at {x[i]}: {y[i]}")
        assert y[i] == f(x[i])

    print(f"Sphere function value at global minimum {f.global_minimum}: {f(f.global_minimum)}")
    assert np.isclose(f(f.global_minimum), 0.0)
