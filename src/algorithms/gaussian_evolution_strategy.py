import numpy as np
from typing import Optional
from tqdm.auto import tqdm

from src.functions._function import Function
from src.algorithms._algorithm import Algorithm, Result


class GaussianEvolutionStrategy(Algorithm):
    """Evolution Strategy algorithm using Gaussian sampling.
    References:
        https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/
        https://blog.otoro.net/2017/10/29/visual-evolution-strategies/
        https://pymoo.org/algorithms/soo/es.html
    """

    @property
    def name(self) -> str:
        return "Evolution Strategy with Gaussian Sampling"

    def __init__(self, *,
                 mean0: np.ndarray = 0.0,
                 sigma0: np.ndarray = 1.0):
        super().__init__()
        self._mean0: np.ndarray = mean0
        self._sigma0: np.ndarray = sigma0
        self._n_func_calls: int = 0

    def optimize(self,
                 function: Function, *,
                 population_size: int,
                 function_dimension: int,
                 initial_population: Optional[np.ndarray] = None,
                 elite_size: Optional[int] = None,
                 keep_elites_between_generations: bool = True,
                 budget: Optional[int] = None,
                 stop_fitness: Optional[float] = None,
                 search_domain: Optional[tuple[float, float]] = None,
                 minimize: bool = True,
                 verbose: bool = False) -> Result:
        """Optimize a given function using Evolution Strategy within a specified budget."""
        assert not (budget is None and stop_fitness is None), \
            "Either budget or stop_fitness must be provided!"
        if elite_size is not None:
            assert elite_size < population_size, "Elite size must be smaller than population size."
        else:
            elite_size = population_size // 4  # Default elite size is 1/4 of population size. I choose this value arbitrarily.
        if initial_population is not None:
            assert initial_population.ndim == 2, "Initial population must be a 2D array."
            assert initial_population.shape[0] <= budget, \
                "Initial population size exceeds budget. You won't have enough budget for even starting your optimization."
            assert initial_population.shape[1] == function_dimension, \
                "Initial population dimension mismatch. Be sure it matches function_dimension argument."

        self._n_func_calls = 0

        # Get search domain bounds
        search_lower_bound, search_upper_bound = function.function_domain if search_domain is None else search_domain
        assert search_lower_bound < search_upper_bound, "Invalid search domain bounds."

        # Initialize population variable
        population = None
        mu, sigma = self._mean0, self._sigma0

        # Variables to keep track of the distribution parameters
        best_fitness_history = []
        mean_fitness_history = []
        population_history = []

        pbar = tqdm(total=budget, desc="ES Progress", disable=not verbose)
        while True:
            # 1. Sample population
            if population is None:
                # First run: Initialize population
                if initial_population is None:
                    # Randomly initialize population within search bounds
                    population = np.random.rand(population_size, function_dimension) * (search_upper_bound - search_lower_bound) + search_lower_bound
                else:
                    population = initial_population
                # Ensure population is within bounds
                population = np.clip(population, search_lower_bound, search_upper_bound)
            else:
                # Subsequent runs: Sample around the mean and std of the elite set
                population = np.random.randn(population_size, function_dimension) * sigma + mu
                population = np.clip(population, search_lower_bound, search_upper_bound)

                # If we are keeping elites between generations, insert them into the new population
                if keep_elites_between_generations:
                    population[:elite_size] = elite_set
            # Record population
            population_history.append(population.copy())

            # 2. Evaluate fitness
            fitness = function(population)
            self._n_func_calls += population.shape[0]
            mean_fitness_history.append(np.mean(fitness))

            # 3. Select elites and discard all other solutions.
            if minimize:
                elite_idx = np.argsort(fitness)[:elite_size]
            else:
                elite_idx = np.argsort(fitness)[-elite_size:]

            elite_set = population[elite_idx]
            fitness_elite_set = fitness[elite_idx]
            best_fitness_history.append(fitness_elite_set[0])

            # 4. Update distribution parameters
            mu = np.mean(elite_set, axis=0)
            sigma = np.std(elite_set, axis=0)

            # Update progress bar
            pbar.update(population.shape[0])
            pbar.set_postfix({"best_fitness": best_fitness_history[-1], "mean_fitness": mean_fitness_history[-1]})

            # 5. Check stopping criteria
            if budget is not None and self._n_func_calls >= budget:
                break
            if stop_fitness is not None:
                if minimize and np.min(fitness) <= stop_fitness:
                    break
                if not minimize and np.max(fitness) >= stop_fitness:
                    break

        pbar.close()

        return {
            "x_opt": elite_set[0],
            "fitness_opt": fitness_elite_set[0],
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
    ges = GaussianEvolutionStrategy()

    result = ges.optimize(
        function=func,
        population_size=40,
        function_dimension=2,
        keep_elites_between_generations=False,
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
