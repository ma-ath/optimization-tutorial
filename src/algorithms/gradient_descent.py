from typing import Optional
import numpy as np
from tqdm.auto import tqdm

from src.functions._function import Function
from src.algorithms._algorithm import Algorithm, Result


class GradientDescent(Algorithm):
    """Gradient Descent optimization algorithm."""

    @property
    def name(self) -> str:
        return "Gradient Descent"

    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        self._lr: float = learning_rate
        self._n_func_calls: int = 0

    def optimize(self,
                 function: Function, *,
                 initial_population: np.ndarray,
                 max_population_size: Optional[int] = None,
                 search_domain: Optional[tuple[float, float]] = None,
                 budget: Optional[int] = None,
                 stop_fitness: Optional[float] = None,
                 minimize: bool = True,
                 function_dimension: int,
                 verbose: bool = False) -> Result:
        """Optimize a given function using Gradient Descent within a specified budget.

        Args:
            function (Function): The objective function to optimize.
            initial_population (np.ndarray): The starting point for the optimization.
            budget (int): The maximum number of function evaluations.
            stop_fitness (Optional[float]): The target fitness value to reach.
            minimize (bool): Whether to minimize or maximize the function.

        Returns:
            Result: The optimization result containing the estimated minimum point.
        """

        self._n_func_calls = 0

        search_lower_bound, search_upper_bound = function.function_domain if search_domain is None else search_domain
        if type(search_lower_bound) is float:
            search_lower_bound = np.full((function_dimension,), search_lower_bound)
        if type(search_upper_bound) is float:
            search_upper_bound = np.full((function_dimension,), search_upper_bound)
        assert len(search_lower_bound) == function_dimension, \
            "search_lower_bound length mismatch. Be sure it matches function_dimension argument."
        assert len(search_upper_bound) == function_dimension, \
            "search_upper_bound length mismatch. Be sure it matches function_dimension argument."
        assert np.all(search_upper_bound > search_lower_bound), \
            "search_upper_bound must be greater than search_lower_bound for all dimensions."

        if max_population_size is not None:
            assert initial_population.shape[0] <= max_population_size, \
                "Initial population size exceeds max_population_size."

        if budget is None and stop_fitness is None:
            self._logger.error("Either budget or stop_fitness must be provided!")
            raise AssertionError("Either budget or stop_fitness must be provided!")

        assert initial_population.shape[initial_population.ndim - 1] == function_dimension, \
            "Initial population dimension mismatch. Be sure it matches function_dimension argument."

        assert function.is_differentiable, "Function must be differentiable for Gradient Descent."

        x = initial_population
        x_history = [x.copy()]
        fitness_history_best = []

        pbar = tqdm(total=budget, desc="Gradient Descent Optimization", disable=not verbose)
        while True:
            # Update position
            if minimize:
                x = x - self._lr * function.gradient(x)
            else:
                x = x + self._lr * function.gradient(x)

            # Count one function call for the gradient evaluation
            self._n_func_calls += 1

            # Ensure x is within search domain
            x = np.clip(x, search_lower_bound, search_upper_bound)
            x_history.append(x.copy())

            # Count one more function call for the fitness evaluation
            fitness = function(x)
            self._n_func_calls += 1
            fitness_history_best.append(fitness)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"best_fitness": fitness})

            if budget is not None and self._n_func_calls >= budget:
                break

            if stop_fitness is not None:
                if (minimize and (fitness <= stop_fitness).any()) or (not minimize and (fitness >= stop_fitness).any()):
                    break

        pbar.close()
        return {
            "x_opt": x,
            "fitness_opt": fitness,
            "x_history": np.array(x_history),
            "fitness_history": {
                "best": np.array(fitness_history_best)
            },
            "used_budget": self._n_func_calls
        }


if __name__ == "__main__":
    from functions import Sphere

    func = Sphere()
    gd = GradientDescent(learning_rate=0.1)

    result = gd.optimize(
        function=func,
        initial_population=np.array([10.0, 10.0, 10.0]),
        budget=100,
        stop_fitness=1e-6,
        minimize=True,
        function_dimension=3,
    )

    print("Optimized x:", result["x_opt"])
    print("Function value at optimized x:", result["fitness_opt"])
    print("Optimization history:", result["x_history"])
    print("Fitness history:", result["fitness_history"]["best"])
