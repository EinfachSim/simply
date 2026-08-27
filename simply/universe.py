import yaml
from .bodies.celestial import CelestialBody
from .gravity import GM_REGISTRY
from .integration import INTEG_REGISTRY
import torch

class Universe:

    def __init__(self, cfg):
        with open(cfg) as stream:
            uni_dict = yaml.safe_load(stream)

        self.bodies, self.system_state = self._parse_celestial_bodies(uni_dict["celestial_bodies"])

        self.gravity_model = self._parse_gravity_model(uni_dict["gravity_model"])

        self.integrator = self._parse_integrator(uni_dict["integrator"])

    def _parse_celestial_bodies(self, body_dict):
        body_list = []
        state = []
        for body_name in body_dict.keys():

            body = body_dict[body_name]
            body_list.append(CelestialBody(body_name, **body))
            state.append(body_list[-1].state())
        
        return body_list, torch.vstack(state)

    def _parse_gravity_model(self, gm_dict):
        gm_type = gm_dict["type"]
        gm_G = gm_dict["G"]
        return GM_REGISTRY[gm_type](gm_G, self.bodies)

    def _parse_integrator(self, integ_str):
        return INTEG_REGISTRY[integ_str](self)

    #state is a matrix of shape (n, 6) where n is the number of bodies
    def derivative(self, state: torch.Tensor, t=0) -> torch.Tensor:
        poss, vels = state[:, :3], state[:, 3:]
        accs = self.acceleration(poss)
        return torch.hstack([vels, accs])

    def acceleration(self, state, t=0) -> torch.Tensor:
        poss = state[:, :3]
        return self.gravity_model(poss)

    def get_state(self) -> torch.Tensor:
        return self.system_state

    def simulate(self, tstart, tend, dt) -> torch.Tensor:

        states = self.integrator.integrate(tstart, tend, dt, self.system_state)
        masses = torch.Tensor([body.mass for body in self.bodies])
        radii = torch.Tensor([body.radius for body in self.bodies])
        colors = [body.color for body in self.bodies]
        names = [body.name for body in self.bodies]

        torch.save({
            "masses": masses,
            "radii": radii,
            "states": states,
            "colors": colors,
            "names": names,
            "tspan": [tstart, tend, dt]
        }, f="traj.pt")

        return states

    #returns the relative velocity (scalar, you decide direction!) needed to orbit
    # around the CelestialBody of name center_name at the radius wanted_radius
    def get_circ_orbit_params(self, wanted_radius, center_name="earth"):
        import numpy as np
        for body in self.bodies:
            if body.name == center_name:
                M = body.mass
        if not M:
            raise Exception(f"Could not find celestial body with name {center_name} in universe!")

        
        v = np.sqrt((self.gravity_model.G()*M)/wanted_radius)

        return v
        