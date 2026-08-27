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
        self.m_dry = sc_dict["m_dry"]
        self.m_fuel = sc_dict["m_fuel"]
        self.mass = self.m_dry + self.m_fuel
        self.v_e = sc_dict["v_e"]

        self._state_vec = torch.tensor(self.pos + self.vel + [self.mass], dtype=torch.float64)


    def _parse_integrator(self, integ_str):
        return INTEG_REGISTRY[integ_str](self)

    def _parse_gravity_model(self, gm_dict):
        gm_type = gm_dict["type"]
        gm_G = gm_dict["G"]
        tensors = torch.load(gm_dict["tensor_path"])
        return GM_REGISTRY[gm_type].from_tensors(gm_G, tensors)

    def state(self):
        return self._state_vec
    def acceleration(self, pos, t, mass=None, thrust=None):
        _pos = pos.squeeze(0)
        acc_g = self.gravity_model.gravity_at(_pos, t)

        acc = acc_g
        if thrust is not None and mass is not None:
            acc_thrust = thrust / mass
            acc += acc_thrust
        return acc

    def derivative(self, state, t=0):
        pos, vel, mass = state[:, :3], state[:, 3:-1], state[:, -1]

        m_dot = torch.zeros(1)
        if self.thrust_fn is not None:
            #Tsiolkovsky mass derivative
            thrust_vec = self.thrust_fn(t)
            thrust_mag = thrust_vec.norm()
            fuel_remaining = mass - self.m_dry
            m_dot = -thrust_mag / (self.v_e)
            m_dot = m_dot * (fuel_remaining > 0)

            acc = self.acceleration(pos, t, mass=mass, thrust=thrust_vec)
        #ballistic case
        else:
            acc = self.acceleration(pos, t, mass=mass)

        return torch.hstack([vel, acc.unsqueeze(0), m_dot.unsqueeze(0)])

    def simulate(self, tstart, tend, dt, thrust_fn = None) -> torch.Tensor:

        self.thrust_fn = thrust_fn

        states = self.integrator.integrate(tstart, tend, dt, self.state().unsqueeze(0))
        return states.squeeze(1)