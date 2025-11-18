from typing import Optional
import numpy as np
from tqdm.auto import tqdm
from cmaes import CMA

from src.functions._function import Function
from src.algorithms._algorithm import Algorithm, Result


class CMAES(Algorithm):
    """Covariance Matrix Adaptation Evolution Strategy optimization algorithm.
    (This title is too long for a class name so let's just call it CMA-ES).

    Since CMA-ES is a bit more complex algorithm, let's wrap the well-known cmaes package from Nomuraさん and Shibataさん.
    References:
        https://github.com/CyberAgentAILab/cmaes
        https://arxiv.org/abs/2402.01373
    """

    @property
    def name(self) -> str:
        return "Covariance Matrix Adaptation - Evolution Strategy (CMA-ES)"

    def __init__(self, *,
                 initial_mean: Optional[np.ndarray] = None,
                 initial_sigma: Optional[float] = None):
        super().__init__()
        self._initial_mean = initial_mean
        self._initial_sigma = initial_sigma
        self._n_func_calls: int = 0

    def optimize(self,
                 function: Function, *,
                 initial_population: Optional[np.ndarray] = None,
                 max_population_size: int,
                 search_domain: Optional[tuple[float, float]] = None,
                 budget: Optional[int] = None,
                 stop_fitness: Optional[float] = None,
                 minimize: bool = True,
                 function_dimension: int,
                 verbose: bool = False) -> Result:
        """Optimize a given function using CMA-ES within a specified budget."""
        assert not (budget is None and stop_fitness is None), \
            "Either budget or stop_fitness must be provided!"
        assert budget is None or max_population_size <= budget, \
            "Population size exceeds budget. You won't have enough budget for even starting your optimization."
        
        search_lower_bound, search_upper_bound = function.function_domain if search_domain is None else search_domain
        assert search_lower_bound < search_upper_bound, "Invalid search domain bounds."

        # Initialize CMA-ES optimizer
        if self._initial_mean is None:
            self._initial_mean = np.zeros(function_dimension)
        if self._initial_sigma is None:
            self._initial_sigma = 1.0

        cma = CMA(
            mean=self._initial_mean,
            sigma=self._initial_sigma,
            bounds=np.array([search_lower_bound, search_upper_bound]).reshape(1, -1).repeat(function_dimension, axis=0),
            max_population_size=max_population_size
        )
        self._n_func_calls = 0
        population = None

        # Variables to keep track of the distribution parameters
        best_fitness_history = []
        mean_fitness_history = []
        population_history = []
        best_solution = None
        best_solution_fitness = float('inf') if minimize else float('-inf')

        pbar = tqdm(total=budget, desc="CMA-ES Progress", disable=not verbose)
        while True:
            # 1. Sample population
            if population is None and initial_population is not None:
                # First run: Initialize population and ensure it is within bounds
                assert initial_population.ndim == 2, "Initial population must be a 2D array."
                assert initial_population.shape[0] <= budget, \
                    "Initial population size exceeds budget. You won't have enough budget for even starting your optimization."
                assert initial_population.shape[1] == function_dimension, \
                    "Initial population dimension mismatch. Be sure it matches function_dimension argument."
                population = initial_population
                population = np.clip(population, search_lower_bound, search_upper_bound)
            else:
                # Subsequent runs: Sample around the mean and std of the elite set
                if budget is None or self._n_func_calls + max_population_size > budget:
                    # Limit evaluations if exceeding max allowed budget
                    max_population_size = budget - self._n_func_calls
                population = np.vstack([cma.ask() for _ in range(max_population_size)])
            population_history.append(population.copy())

            # 2. Evaluate fitness
            fitness = function(population)
            self._n_func_calls += population.shape[0]
            mean_fitness_history.append(np.mean(fitness))
            best_fitness_history.append(np.min(fitness) if minimize else np.max(fitness))

            if minimize and best_fitness_history[-1] < best_solution_fitness:
                best_solution_fitness = best_fitness_history[-1]
                best_solution = population[np.argmin(fitness)]
            elif not minimize and best_fitness_history[-1] > best_solution_fitness:
                best_solution_fitness = best_fitness_history[-1]
                best_solution = population[np.argmax(fitness)]

            # 3. Update CMA-ES distributions
            solutions = [(x_sample, fitness_val) for x_sample, fitness_val in zip(population, fitness)]
            cma.tell(solutions)

            # Update progress bar
            pbar.update(population.shape[0])
            pbar.set_postfix({"best_fitness": best_fitness_history[-1], "mean_fitness": mean_fitness_history[-1]})

            # 4. Check stopping criteria
            if budget is not None and self._n_func_calls >= budget:
                break
            if stop_fitness is not None:
                if minimize and np.min(fitness) <= stop_fitness:
                    break
                if not minimize and np.max(fitness) >= stop_fitness:
                    break

        pbar.close()

        return {
            "x_opt": best_solution,
            "fitness_opt": best_solution_fitness,
            "x_history": np.array(population_history),
            "fitness_history": {
                "best": np.array(best_fitness_history),
                "mean": np.array(mean_fitness_history)
            },
            "used_budget": self._n_func_calls
        }


if __name__ == "__main__":
    from functions import Sphere

    func = Sphere()
    cma = CMAES(initial_mean=np.array([0.0, 0.0]), initial_sigma=0.5)

    result = cma.optimize(
        function=func,
        max_population_size=40,
        function_dimension=2,
        budget=1000,
        stop_fitness=None,
        minimize=True,
        verbose=True
    )

    print("Best solution found:", result["x_opt"])
    print("Best fitness achieved:", result["fitness_opt"])
    print("Total function evaluations:", result["used_budget"])
    print("Fitness history (best):", result["fitness_history"]["best"])
    print("Fitness history (mean):", result["fitness_history"]["mean"])
    print("Population history shape:", result["x_history"].shape)
