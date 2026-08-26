from abc import ABC, abstractmethod

class BaseIntegrator(ABC):

    @property
    @abstractmethod
    def step(self, dt):
        ...
