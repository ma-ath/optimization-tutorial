import numpy as np
from typing import Optional, Any
from cocoex import Problem

from src.algorithms._algorithm import Algorithm
from src.functions._function import Function


class CocoProblemWrapper(Function):
    """Wrapper to run an Algorithm on a COCO problem."""

    @property
    def name(self) -> str:
        return f"{self.problem.name}"
    
    @property
    def function_domain(self) -> tuple[float, float]:
        return self.problem.lower_bounds[0], self.problem.upper_bounds[0]

    def __init__(self,
                 problem: Problem,
                 x0: Optional[np.ndarray] = None) -> None:
        self.problem = problem
        self.x0 = x0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_coco = []
        for i in range(x.shape[0]):
            x_coco.append(self.problem(x[i]))
        return np.array(x_coco)


if __name__ == "__main__":
    from cocoex import Suite
    from src.algorithms import DifferentialEvolution
    suite = Suite(
        suite_name="bbob", 
        suite_instance="year: 2009", 
        suite_options="function_indices: 1 instance_indices: 1-10 dimensions: 2,20"
    )
    problem = suite[0]
    wrapper = CocoProblemWrapper(problem)
    algorithm = DifferentialEvolution()

    result = algorithm.optimize(wrapper, max_population_size=10, function_dimension=2, budget=1000, verbose=True)
    print("Best solution found:", result["x_opt"])
    print("Function value at best solution:", result["fitness_opt"])
