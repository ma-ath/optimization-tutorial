import numpy as np
from typing import TypedDict

from functions._function import Function


class Result(TypedDict):
    x_min: np.ndarray


class Algorithm:
    """Base class for optimization algorithms."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def optimize(self,
                 function: Function,
                 budget: int,
                 minimize: bool = True) -> Result:
        """Optimize a given function within a specified budget.
        
        Args:
            function (Function): The objective function to optimize.
            minimize (bool): Whether to minimize or maximize the function.

        Returns:
            Result: The optimization result.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
