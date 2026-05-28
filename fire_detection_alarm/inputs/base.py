from abc import ABC, abstractmethod
import numpy as np

class BaseSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...
