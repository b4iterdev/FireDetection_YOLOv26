from abc import ABC, abstractmethod

class BaseSource(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def release(self):
        pass
