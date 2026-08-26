from .base import BaseIntegrator
import torch

class RK4Integrator(BaseIntegrator):

    def __init__(self, system):

        self.system = system


    def step(self, dt, state, t=0):

        k1 = self.system.derivative(state, t)
        k2 = self.system.derivative(state + k1 * dt/2, t + dt/2)
        k3 = self.system.derivative(state + k2 * dt/2, t + dt/2)
        k4 = self.system.derivative(state + k3 * dt, t + dt)

        state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return state

    def integrate(self, tstart, tend, dt, state, debug = True):
        tspan = torch.arange(start =tstart, end=tend, step=dt)
        states = [state]

        for t in tspan:
            if debug:
                print(f"[Integ] t = {t}")
            state = self.step(dt, state, t)
            states.append(state)

        states = torch.stack(states)
        return states
        