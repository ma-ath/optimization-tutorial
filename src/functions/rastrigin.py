import numpy as np

from _function import Function


class Rastrigin(Function):
    r"""Rastrigin function.
        f(\bm{x}) = A n + \sum_{i=1}^{n} \left(x_i^2 - A \cos(2 \pi x_i)\right), \bm{x} \in \mathbb{R}^n.

    Global minimum at x_{min} = [0, 0, ..., 0], f(x_{min}) = 0.
    """

    @property
    def global_minimum(self) -> np.ndarray:
        return np.array([0.0, 0.0])
    
    @property
    def global_maximum(self) -> np.ndarray:
        return np.array([[4.52299366, 4.52299366],
                         [-4.52299366, -4.52299366],
                         [4.52299366, -4.52299366],
                         [-4.52299366, 4.52299366]])
    
    @property
    def search_domain(self) -> tuple[float, float]:
        return (-5.12, 5.12)

    def __init__(self, A: float = 10.0):
        super().__init__()
        self._A = A

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
    x = np.random.uniform(f.search_domain[0], f.search_domain[1], (5, 3))
    y = f(x)
    for i in range(x.shape[0]):
        assert y[i] == f(x[i])
        print(f"Rastrigin function value at {x[i]}: {y[i]}")

    print(f"Rastrigin function value at global minimum {f.global_minimum}: {f(f.global_minimum)}")
    assert np.isclose(f(f.global_minimum), 0.0)

    for gm in f.global_maximum:
        print(f"Rastrigin function value at global maximum {gm}: {f(gm)}")
        assert np.isclose(f(gm), 80.70658039)
