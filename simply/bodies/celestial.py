from .base import BasePhysicalBody
import torch

class CelestialBody(BasePhysicalBody):

    def __init__(self, name, **kwargs):

        self.pos = kwargs["pos"]
        self.vel = kwargs["vel"]
        self.mass = kwargs["mass"]
        self.radius = kwargs["radius"]
        self.color = kwargs["color"]
        self.name = name

        self._state_vec = torch.tensor(self.pos + self.vel, dtype=torch.float64)

    def state(self):
        return self._state_vec

    def derivative(self, t):
        ...
