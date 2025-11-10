from typing import Optional
import numpy as np
from tqdm.auto import tqdm

from functions._function import Function
from _algorithm import Algorithm, Result


class GradientDescent(Algorithm):
    """Gradient Descent optimization algorithm."""

    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        self._lr: float = learning_rate
        self._n_func_calls: int = 0

    def optimize(self,
                 function: Function, *,
                 initial_guess: np.ndarray,
                 budget: Optional[int] = None,
                 stop_fitness: Optional[float] = None,
                 minimize: bool = True) -> Result:
        """Optimize a given function using Gradient Descent within a specified budget.
        
        Args:
            function (Function): The objective function to optimize.
            initial_guess (np.ndarray): The starting point for the optimization.
            budget (int): The maximum number of function evaluations.
            stop_fitness (Optional[float]): The target fitness value to reach.
            minimize (bool): Whether to minimize or maximize the function.

        Returns:
            Result: The optimization result containing the estimated minimum point.
        """

        if budget is None and stop_fitness is None:
            self._logger.error("Either budget or stop_fitness must be provided!")
            raise AssertionError("Either budget or stop_fitness must be provided!")

        x = initial_guess
        x_history = [x.copy()]

        pbar = tqdm(total=budget, desc="Gradient Descent Optimization")
        while True:
            # Update position
            if minimize:
                x = x - self._lr * function.gradient(x)
            else:
                x = x + self._lr * function.gradient(x)

            # Count the number of function evaluations
            self._n_func_calls += 1
            x_history.append(x.copy())

            fitness = function(x)
            pbar.update(1)

            if budget is not None and self._n_func_calls >= budget:
                break

            if stop_fitness is not None:
                if (minimize and fitness <= stop_fitness) or (not minimize and fitness >= stop_fitness):
                    break

        pbar.close()
        return {
            "x_opt": x,
            "x_history": x_history
        }


if __name__ == "__main__":
    # Use package-relative import so this module can be executed as
    # `python -m src.algorithms.gradient_descent` or run from the package.
    from functions import Sphere

    func = Sphere()
    gd = GradientDescent(learning_rate=0.1)

    result = gd.optimize(
        function=func,
        initial_guess=np.array([10.0, 10.0, 10.0]),
        budget=100,
        stop_fitness=1e-6,
        minimize=True
    )

    print("Optimized x:", result["x_opt"])
    print("Function value at optimized x:", func(result["x_opt"]))
    print("Optimization history:", result["x_history"])
