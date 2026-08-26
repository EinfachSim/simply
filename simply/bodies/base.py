from abc import ABC, abstractmethod
class BasePhysicalBody(ABC):

    @property
    @abstractmethod
    def state(self):
        ...

    @property
    @abstractmethod
    def derivative(self, t, state):
        ...

