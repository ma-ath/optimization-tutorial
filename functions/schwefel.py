import numpy as np
from _function import Function


class Schwefel(Function):
    r"""Schwefel function.
        f(\bm{x}) = 418.9829n - \sum_{i=1}^{n} x_i \sin(\sqrt{|x_i|}), \bm{x} \in \mathbb{R}^n.

    Global minimum at x_{min} = [420.9687, 420.9687, ..., 420.9687], f(x_{min}) = 0.
    """

    @property
    def global_minimum(self) -> np.ndarray:
        return np.array([420.9687, 420.9687])
    
    @property
    def search_domain(self) -> tuple[float, float]:
        return (-500.0, 500.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Schwefel function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Function value(s) at the input point(s).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return 418.9829*x.shape[1] - np.sum(x*np.sin(np.sqrt(np.abs(x))), axis=1)


if __name__ == "__main__":
    f = Schwefel()
    x = np.random.uniform(f.search_domain[0], f.search_domain[1], (5, 3))
    y = f(x)
    for i in range(x.shape[0]):
        print(f"Schwefel function value at {x[i]}: {y[i]}")
        assert y[i] == f(x[i])

    print(f"Schwefel function value at global minimum {f.global_minimum}: {f(f.global_minimum)}")
    assert np.isclose(f(f.global_minimum), 0.0, atol=1e-4)
