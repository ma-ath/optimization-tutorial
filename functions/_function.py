import numpy as np


class Function:
    """
    Base class for mathematical functions.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def search_domain(self) -> tuple[float, float]:
        return (-np.inf, np.inf)
    
    @property
    def global_maximum(self) -> np.ndarray:
        raise UndefinedValue("This function does not have a global maximum defined.")
    
    @property
    def global_minimum(self) -> np.ndarray:
        raise UndefinedValue("This function does not have a global minimum defined.")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")


class UndefinedValue(Exception):
    """Raised when a variable, value, or state is undefined."""
    def __init__(self, message: str = None):
        super().__init__(message or "Value is undefined.")
