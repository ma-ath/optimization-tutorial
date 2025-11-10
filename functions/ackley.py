import numpy as np
from _function import Function


class Ackley(Function):
    r"""Ackley function.
        f(\bm{x}) = -a \exp\left(-b \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}\right) - \exp\left(\frac{1}{n} \sum_{i=1}^{n} \cos(c x_i)\right) + a + e,
        \bm{x} \in \mathbb{R}^n.

    Global minimum at x_{min} = [0, 0, ..., 0], f(x_{min}) = 0.
    """

    @property
    def global_minimum(self) -> np.ndarray:
        return np.array([0.0, 0.0])
    
    @property
    def search_domain(self) -> tuple[float, float]:
        return (-32.768, 32.768)

    def __init__(self, a: float = 20, b: float = 0.2, c: float = 2 * np.pi):
        self._a = a
        self._b = b
        self._c = c
        super().__init__()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Ackley function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Function value(s) at the input point(s).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        n = x.shape[1]
        sum1 = np.sum(x**2, axis=1)
        sum2 = np.sum(np.cos(self._c * x), axis=1)
        return -self._a * np.exp(-self._b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + self._a + np.e


if __name__ == "__main__":
    f = Ackley()
    x = np.random.uniform(f.search_domain[0], f.search_domain[1], (5, 3))
    y = f(x)
    for i in range(x.shape[0]):
        assert y[i] == f(x[i])
        print(f"Ackley function value at {x[i]}: {y[i]}")

    assert np.isclose(f(f.global_minimum), 0.0)
    print(f"Ackley function value at global minimum {f.global_minimum}: {f(f.global_minimum)}")
