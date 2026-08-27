from .base import BaseIntegrator
import torch

class VerletIntegrator(BaseIntegrator):

    def __init__(self, system):

        self.system = system


    def step(self, dt, state, t=0):
        poss, vels = state[:, :3], state[:, 3:]

        accs = self.system.acceleration(poss, t)

        vels_half = vels + 0.5 * dt * accs
        poss_new = poss + dt * vels_half
        accs_new = self.system.acceleration(poss_new, t + dt)
        vels_new = vels_half + 0.5 * dt * accs_new

        return torch.hstack([poss_new, vels_new])

    def integrate(self, tstart, tend, dt, state, debug = True):
        tspan = torch.arange(start =tstart, end=tend, step=dt, dtype=torch.float64)
        states = [state]

        for t in tspan:
            if debug:
                print(f"[Integ] t = {t}")
            state = self.step(dt, state, t)
            states.append(state)

        states = torch.stack(states)
        return states
        