from typing import Optional
import numpy as np
from tqdm.auto import tqdm

from src.functions._function import Function
from src.algorithms._algorithm import Algorithm, Result


class DifferentialEvolution(Algorithm):
    """Differential Evolution optimization algorithm."""
    def __init__(self, *,
                 F: float = 0.5,
                 CR: float = 0.7):
        super().__init__()
        self._F: float = F
        self._CR: float = CR
        self._n_func_calls: int = 0

    def optimize(self,
                 function: Function, *,
                 population_size: int,
                 function_dimension: int,
                 initial_population: Optional[np.ndarray] = None,
                 budget: Optional[int] = None,
                 stop_fitness: Optional[float] = None,
                 minimize: bool = True,
                 verbose: bool = False) -> Result:
        """Optimize a given function using Differential Evolution within a specified budget."""
        assert population_size >= 4, "Population size must be at least 4."
        assert not (budget is None and stop_fitness is None), \
            "Either budget or stop_fitness must be provided!"

        self._n_func_calls = 0

        search_lower_bound, search_upper_bound = function.search_domain

        # Randomly initialize population
        if initial_population is None:
            population = np.random.rand(population_size, function_dimension) * (search_upper_bound - search_lower_bound) + search_lower_bound
        else:
            assert initial_population.shape == (population_size, function_dimension), \
                "Initial population shape mismatch. Be sure it matches (population_size, function_dimension)."
            assert budget is None or initial_population.shape[0] <= budget, \
                "Initial population size exceeds budget. You won't have enough budget for even starting your optimization."
            population = initial_population

        # Compute the fitness of the initial population
        fitness = function(population)
        self._n_func_calls += population.shape[0]

        best_fitness_history = []
        mean_fitness_history = []
        population_history = [population.copy()]

        # Optimization loop
        pbar = tqdm(total=budget, desc="DE Progress", disable=not verbose)
        while True:
            # Pick three *distinct* indices not equal itself.
            idxs = np.arange(population_size).reshape(1, -1).repeat(population_size, axis=0)
            mask = np.eye(population_size, dtype=bool)
            idxs = idxs[~mask].reshape(population_size, population_size - 1)

            permutations = np.vstack([
                np.random.permutation(population_size - 1)[:3]
                for _ in range(population_size)
            ])
            r = idxs.take(permutations)
            assert r.shape[1] >= 3
            candidates = population[r]  # Shape: (n_population, 3, dim)

            # Mutation
            mutants = candidates[:, 0] + self._F * (candidates[:, 1] - candidates[:, 2])
            mutants = np.clip(mutants, search_lower_bound, search_upper_bound)

            # Crossover (binomial)
            cross_points = np.random.rand(population_size, function_dimension) < self._CR
            j_rand = np.random.randint(0, function_dimension, (population_size,))
            cross_points[np.arange(population_size), j_rand] = True
            trials = np.where(cross_points, mutants, population)

            # Selection
            if budget is None or self._n_func_calls + trials.shape[0] <= budget:
                f_trials = function(trials)
                self._n_func_calls += f_trials.shape[0]
            else:
                # Limit evaluations if exceeding max allowed
                f_trials = function(trials[:budget - self._n_func_calls])
                self._n_func_calls += f_trials.shape[0]
                f_trials = np.pad(f_trials, (0, trials.shape[0] - f_trials.shape[0]),
                        value=float('inf') if not minimize else float('-inf'))
                    
            if minimize:
                better_mask = f_trials < fitness
            else:
                better_mask = f_trials > fitness

            population = np.where(better_mask[:, None], trials, population)
            fitness = np.where(better_mask, f_trials, fitness)

            # Record best and mean fitness
            best_idx = np.argmin(fitness) if minimize else np.argmax(fitness)
            best_fitness_history.append(fitness[best_idx].item())
            mean_fitness_history.append(fitness.mean().item())
            population_history.append(population.copy())

            # Update progress bar
            pbar.set_postfix({"best_fitness": best_fitness_history[-1], "mean_fitness": mean_fitness_history[-1]})
            pbar.update(trials.shape[0])

            # Check stopping criteria
            if budget is not None and self._n_func_calls >= budget:
                break
 
            if stop_fitness is not None:
                if (minimize and fitness[best_idx] <= stop_fitness) or (not minimize and fitness[best_idx] >= stop_fitness):
                    break

        pbar.close()

        return {
            "x_opt": population[best_idx],
            "fitness_opt": fitness[best_idx],
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
    de = DifferentialEvolution()

    result = de.optimize(
        function=func,
        population_size=20,
        function_dimension=10,
        initial_population=None,
        budget=1000,
        stop_fitness=None,
        minimize=True,
        verbose=True
    )
