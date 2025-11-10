import numpy as np


class Function:
    """
    Base class for mathematical functions.
    """
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def search_domain(self) -> tuple:
        raise NotImplementedError("Subclasses must implement this property")
    
    @property
    def global_maximum(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this property")
    
    @property
    def global_minimum(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this property")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")
