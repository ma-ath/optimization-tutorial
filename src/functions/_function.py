import numpy as np
from typing import Union


class Function:
    """
    Base class for mathematical functions.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def function_domain(self) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        return (-np.inf, np.inf)

    @property
    def global_maximum(self) -> np.ndarray:
        raise UndefinedValue("This function does not have a global maximum defined.")

    @property
    def global_minimum(self) -> np.ndarray:
        raise UndefinedValue("This function does not have a global minimum defined.")

    @property
    def is_differentiable(self) -> bool:
        return False

    @property
    def is_convex(self) -> bool:
        return False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise UndefinedValue("Gradient is not defined for this function.")


class UndefinedValue(Exception):
    """Raised when a variable, value, or state is undefined."""
    def __init__(self, message: str = None):
        super().__init__(message or "Value is undefined.")
