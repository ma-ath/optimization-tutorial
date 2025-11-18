import numpy as np
import logging
from typing import TypedDict, NotRequired, Optional

from src.functions._function import Function


class Result(TypedDict):
    class FitnessHistory(TypedDict):
        best: np.ndarray
        mean: NotRequired[np.ndarray]

    x_opt: np.ndarray
    fitness_opt: float
    x_history: NotRequired[list[np.ndarray]]
    fitness_history: NotRequired[FitnessHistory]
    used_budget: NotRequired[int]


class Algorithm:
    """Base class for optimization algorithms."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def optimize(self,
                 function: Function, *,
                 initial_population: Optional[np.ndarray] = None,
                 max_population_size: Optional[int] = None,
                 search_domain: Optional[tuple[float, float]] = None,
                 budget: Optional[int] = None,
                 stop_fitness: Optional[float] = None,
                 minimize: bool = True) -> Result:
        """Optimize a given function within a specified budget.
        
        Args:
            function (Function): The objective function to optimize.
            budget (Optional[int]): The maximum number of function evaluations.
                if None, there is no limit on evaluations.
            stop_fitness (Optional[float]): The target fitness value to reach.
                if not None, the optimization will halt when this fitness is achieved.
            minimize (bool): Whether to minimize or maximize the function.

        Returns:
            Result: The optimization result.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
