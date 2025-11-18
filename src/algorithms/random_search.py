from typing import Optional
import numpy as np
from tqdm.auto import tqdm

from src.functions._function import Function
from src.algorithms._algorithm import Algorithm, Result


class RandomSearch(Algorithm):
    """Random Search optimization algorithm."""

    @property
    def name(self) -> str:
        return "Random Search"

    def __init__(self):
        super().__init__()
        self._n_func_calls: int = 0

    def optimize(self,
                 function: Function, *,
                 initial_population: Optional[np.ndarray] = None,
                 max_population_size: int = 20,
                 search_domain: Optional[tuple[float, float]] = None,
                 budget: Optional[int] = None,
                 stop_fitness: Optional[float] = None,
                 minimize: bool = True,
                 function_dimension: int,
                 verbose: bool = False) -> Result:
        """Optimize the given function using Random Search.
        Args:
            function (Function): The objective function to optimize.
            initial_population (Optional[np.ndarray], optional): The starting point for the optimization. Defaults to None.
            budget (Optional[int], optional): Maximum number of function evaluations. Defaults to None.
            stop_fitness (Optional[float], optional): Target fitness value to stop optimization. Defaults to None.
            minimize (bool, optional): Whether to minimize or maximize the function. Defaults to True.
            verbose (bool, optional): Whether to display progress bar. Defaults to False.

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

        if budget is None and stop_fitness is None:
            self._logger.error("Either budget or stop_fitness must be provided!")
            raise AssertionError("Either budget or stop_fitness must be provided!")

        if initial_population is not None:
            if initial_population.ndim == 1:
                initial_population = initial_population[np.newaxis, :]
            if initial_population.shape[1] != function_dimension:
                self._logger.error(f"Initial guess has incorrect dimension {initial_population.shape[1]}, expected {function_dimension}")
                raise ValueError(f"Initial guess has incorrect dimension {initial_population.shape[1]}, expected {function_dimension}")

        x_history = []
        fitness_history_best = []
        fitness_history_mean = []
        best_solution = None
        best_solution_fitness = float('inf') if minimize else float('-inf')

        pbar = tqdm(total=budget, desc="Random Search Optimization", disable=not verbose)
        while True:
            # Randomly sample new candidate solutions
            if initial_population is not None:
                x = initial_population
                initial_population = None  # Use initial guess only once
            else:
                if budget is not None and self._n_func_calls + max_population_size > budget:
                    # Ensures we do not exceed the budget of function calls
                    max_population_size = budget - self._n_func_calls
                x = np.random.uniform(
                    low=search_lower_bound,
                    high=search_upper_bound,
                    size=(max_population_size, function_dimension)
                )
            x_history.append(x.copy())

            # Compute fitness for all candidates
            fitness = function(x)
            self._n_func_calls += x.shape[0]
            fitness_history_mean.append(np.mean(fitness))

            # Update best solution found so far
            if minimize:
                current_best_idx = np.argmin(fitness)
                if fitness[current_best_idx] < best_solution_fitness:
                    best_solution_fitness = fitness[current_best_idx]
                    best_solution = x[current_best_idx].copy()
            else:
                current_best_idx = np.argmax(fitness)
                if fitness[current_best_idx] > best_solution_fitness:
                    best_solution_fitness = fitness[current_best_idx]
                    best_solution = x[current_best_idx].copy()
            fitness_history_best.append(best_solution_fitness)

            # Update progress bar
            pbar.update(x.shape[0])
            pbar.set_postfix({"best_fitness": best_solution_fitness})

            if budget is not None and self._n_func_calls >= budget:
                break

            if stop_fitness is not None:
                if (minimize and best_solution_fitness <= stop_fitness) or (not minimize and best_solution_fitness >= stop_fitness):
                    break

        pbar.close()
        return {
            "x_opt": best_solution,
            "fitness_opt": best_solution_fitness,
            "x_history": np.array(x_history),
            "fitness_history": {
                "best": np.array(fitness_history_best),
                "mean": np.array(fitness_history_mean)
            },
            "used_budget": self._n_func_calls
        }


if __name__ == "__main__":
    from functions import Sphere

    func = Sphere()
    ges = RandomSearch()

    result = ges.optimize(
        function=func,
        max_population_size=40,
        function_dimension=2,
        budget=1000,
        stop_fitness=None,
        minimize=True,
        verbose=True,
        search_domain=(-1.0, 1.0)
    )

    print("Best solution found:", result["x_opt"])
    print("Best fitness achieved:", result["fitness_opt"])
    print("Total function evaluations:", result["used_budget"])
    print("Fitness history (best):", result["fitness_history"]["best"])
    print("Fitness history (mean):", result["fitness_history"]["mean"])
    print("Population history shape:", result["x_history"].shape)
