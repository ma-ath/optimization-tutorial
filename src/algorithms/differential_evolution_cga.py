import numpy as np
from itertools import product

from typing import Optional
import numpy as np
from tqdm.auto import tqdm

from src.functions._function import Function
from src.algorithms._algorithm import Algorithm, Result


class DifferentialEvolutionCGA(Algorithm):
    """Differential Evolution with CGA crossover selection."""

    @property
    def name(self) -> str:
        return "Differential Evolution with CGA(DE-CGA)"

    def __init__(self, *,
                 F: float = 0.5,
                 CR: float = 0.7):
        super().__init__()
        self._F: float = F
        self._CR: float = CR
        self._n_func_calls: int = 0

    def optimize(self,
                 function: Function, *,
                 initial_population: Optional[np.ndarray] = None,
                 max_population_size: int = 25,
                 search_domain: Optional[tuple[float, float]] = None,
                 budget: Optional[int] = None,
                 stop_fitness: Optional[float] = None,
                 minimize: bool = True,
                 function_dimension: int,
                 verbose: bool = False) -> Result:
        """Optimize a given function using Differential Evolution within a specified budget."""
        assert round(np.sqrt(max_population_size)) == np.sqrt(max_population_size), "Population size must be a square number"
        assert not (budget is None and stop_fitness is None), \
            "Either budget or stop_fitness must be provided!"

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

        # Randomly initialize population
        if initial_population is None:
            assert budget is None or max_population_size <= budget, \
                "Initial population size exceeds budget. You won't have enough budget for even starting your optimization."
            population = np.random.rand(max_population_size, function_dimension) * (search_upper_bound - search_lower_bound) + search_lower_bound
        else:
            assert initial_population.shape[0] >= 4, "Initial population size must be at least 4."
            assert initial_population.shape[1] == function_dimension, \
                "Initial population dimension does not match the specified function dimension."
            assert budget is None or initial_population.shape[0] <= budget, \
                "Initial population size exceeds budget. You won't have enough budget for even starting your optimization."
            population = initial_population

        # Compute the fitness of the initial population
        fitness = function(population)
        self._n_func_calls += population.shape[0]

        best_fitness_history = []
        mean_fitness_history = []
        population_history = [population.copy()]

        # Exchange of materials between cells:
        pop_sqrt = int(np.sqrt(max_population_size))
        cell_can_have_sex_with = np.zeros((max_population_size, max_population_size)).astype(bool)
        for count, (i, j) in enumerate(product(range(pop_sqrt), repeat=2)):
            pivot = i*pop_sqrt+j
            l, r, u, d = pivot-1, pivot+1, pivot-pop_sqrt, pivot+pop_sqrt
            dul, dur, ddl, ddr = pivot-pop_sqrt-1, pivot-pop_sqrt+1, pivot+pop_sqrt+1, pivot+pop_sqrt-1
            for neighborn in [u, d]:
                # Check if neighbount is in population range (no border looping)
                if 0 <= neighborn < max_population_size:  # Only need to check if exceeds vector size
                    cell_can_have_sex_with[count, neighborn] = True
            for neighborn in [l, r]:
                # Check if neighbount is in population range (no border looping)
                # Also need to check if they are in the same line
                if 0 <= neighborn < max_population_size and int(neighborn / pop_sqrt) == int(pivot / pop_sqrt):
                    cell_can_have_sex_with[count, neighborn] = True
            for neighborn in [dul, dur]:
                # Check if neighbount is in population range (no border looping)
                # Also need to check if they are in the previous line
                if 0 <= neighborn < max_population_size and int(neighborn / pop_sqrt) == int(pivot / pop_sqrt) - 1:  # Also need to check if they are in the same line
                    cell_can_have_sex_with[count, neighborn] = True
            for neighborn in [ddl, ddr]:
                # Check if neighbount is in population range (no border looping)
                # Also need to check if they are in the next line
                if 0 <= neighborn < max_population_size and int(neighborn / pop_sqrt) == int(pivot / pop_sqrt) + 1:  # Also need to check if they are in the same line
                    cell_can_have_sex_with[count, neighborn] = True

        # Optimization loop
        pbar = tqdm(total=budget, desc="DE Progress", disable=not verbose)
        while True:
            # Pick three *distinct* indices from neightborn cells, not equal itself.
            idxs = np.arange(max_population_size).reshape(1, -1).repeat(max_population_size, axis=0)
            idxs = np.where(cell_can_have_sex_with, idxs, -1)

            r = np.array([
                np.random.choice(row[row != -1], size=3, replace=False)
                for row in idxs
            ])
            assert r.shape[1] >= 3
            candidates = population[r]  # Shape: (n_population, 3, dim)

            # Mutation
            mutants = candidates[:, 0] + self._F * (candidates[:, 1] - candidates[:, 2])
            mutants = np.clip(mutants, search_lower_bound, search_upper_bound)

            # Crossover (binomial)
            cross_points = np.random.rand(max_population_size, function_dimension) < self._CR
            j_rand = np.random.randint(0, function_dimension, (max_population_size,))
            cross_points[np.arange(max_population_size), j_rand] = True
            trials = np.where(cross_points, mutants, population)

            # Selection
            f_trials = function(trials)
            self._n_func_calls += f_trials.shape[0]
                    
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
    from src.functions import Sphere

    func = Sphere()
    de = DifferentialEvolutionCGA()

    result = de.optimize(
        function=func,
        max_population_size=25,
        function_dimension=10,
        initial_population=None,
        budget=1000,
        stop_fitness=None,
        minimize=True,
        verbose=True
    )
