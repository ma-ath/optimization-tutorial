import numpy as np

from _function import Function


class Himmelblau(Function):
    r"""Himmelblau function.
        f(\bm{x}) = \sum_{i=1}^{n} x_i^2, \bm{x} \in \mathbb{R}^n.
    or equivalently,
        f(\bm{x}) = \bm{x}^T \bm{x}.

    Global minimum at x_{min} = [0, 0, ..., 0], f(x_{min}) = 0.
    """

    @property
    def global_maximum(self) -> np.ndarray:
        return np.array([-0.270845, -0.923039])

    @property
    def global_minimum(self) -> np.ndarray:
        return np.array([[3.0, 2.0],
                         [-2.805118, 3.131312],
                         [-3.779310, -3.283186],
                         [3.584428, -1.848126]])
    
    @property
    def search_domain(self) -> tuple[float, float]:
        return (-5.0, 5.0)

    @property
    def is_differentiable(self) -> bool:
        return True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Himmelblau function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Function value(s) at the input point(s).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[-1] != 2:
            raise ValueError("Himmelblau function is only defined for 2-dimensional input.")
        x1, x2 = x[:, 0], x[:, 1]
        return  (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the Himmelblau function at a given point.
        Args:
            x (np.ndarray): Input array of shape (..., n), where n is the number of dimensions.
        Returns:
            np.ndarray: Gradient vector(s) at the input point(s).
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[-1] != 2:
            raise ValueError("Himmelblau function is only defined for 2-dimensional input.")
        x1, x2 = x[:, 0], x[:, 1]
        a = x1**2 + x2 - 11
        b = x1 + x2**2 - 7
        df_dx1 = 4 * x1 * a + 2 * b
        df_dx2 = 2 * a + 4 * x2 * b
        return np.array([df_dx1, df_dx2]).T


if __name__ == "__main__":
    f = Himmelblau()
    x = np.random.uniform(f.search_domain[0], f.search_domain[1], (5, 2))
    y = f(x)
    for i in range(x.shape[0]):
        print(f"Himmelblau function value at {x[i]}: {y[i]}")
        assert y[i] == f(x[i])

    for gm in f.global_minimum:
        print(f"Himmelblau function value at global minimum {gm}: {f(gm)}")
        assert np.isclose(f(gm), 0.0)

    print(f"Himmelblau function value at global maximum {f.global_maximum}: {f(f.global_maximum)}")
    assert np.isclose(f(f.global_maximum), 181.617)
