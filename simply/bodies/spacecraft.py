from .base import BasePhysicalBody
from ..integration import INTEG_REGISTRY
from ..gravity import GM_REGISTRY
import torch
import yaml

class SpaceCraftBody(BasePhysicalBody):

    def __init__(self, cfg):

        with open(cfg) as stream:
            sc_dict = yaml.safe_load(stream)

        self.gravity_model = self._parse_gravity_model(sc_dict["gravity_model"])
        self.integrator = self._parse_integrator(sc_dict["integrator"])

        self.pos = sc_dict["pos"]
        self.vel = sc_dict["vel"]
        self.mass = sc_dict["mass"]

        self._state_vec = torch.tensor(self.pos + self.vel, dtype=torch.float64)


    def _parse_integrator(self, integ_str):
        return INTEG_REGISTRY[integ_str](self)

    def _parse_gravity_model(self, gm_dict):
        gm_type = gm_dict["type"]
        gm_G = gm_dict["G"]
        tensors = torch.load(gm_dict["tensor_path"])
        return GM_REGISTRY[gm_type].from_tensors(gm_G, tensors)

    def state(self):
        return self._state_vec
    def acceleration(self, pos, t):
        _pos = pos.squeeze(0)
        return self.gravity_model.gravity_at(_pos, t)

    def derivative(self, state, t=0):
        pos, vel = state[:, :3], state[:, 3:]
        acc = self.acceleration(pos, t)
        return torch.hstack([vel, acc.unsqueeze(0)])

    def simulate(self, tstart, tend, dt) -> torch.Tensor:
        states = self.integrator.integrate(tstart, tend, dt, self.state().unsqueeze(0))
        return states.squeeze(1)