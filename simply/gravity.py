from abc import ABC, abstractmethod
import torch
from .utils import get_interp_positions

class BaseGravityModel(ABC):

    @property
    @abstractmethod
    def G(self):
        ...

    #Should return the accelerations
    @abstractmethod
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        ...

class QuadraticGravityModel(BaseGravityModel):

    def __init__(self, G, bodies=None, online=True):
        self._G = G
        self.online = online
        if self.online:
            self.set_bodies(bodies)

    @staticmethod
    def from_tensors(G, tensors):
        gm = QuadraticGravityModel(G, online=False)
        gm.tensors = tensors["states"]
        gm.masses = tensors["masses"]
        gm.radii = tensors["radii"]
        gm.tspan = torch.arange(tensors["tspan"][0], tensors["tspan"][1], step=tensors["tspan"][2], dtype=torch.float64).tolist()
        return gm
    
    def set_bodies(self, bodies):
        self.masses = []
        self.radii = []
        for body in bodies:
            self.masses.append(body.mass)
            self.radii.append(body.radius)

        self.masses = torch.tensor(self.masses, dtype=torch.float64)
        self.radii = torch.tensor(self.radii, dtype=torch.float64)

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        if not self.online:
            raise Exception("Gravity Model cannot be called in offline mode!")
        # Pairwise displacements: R[i,j] = pos[j] - pos[i]
        R = positions[None, :, :] - positions[:, None, :]   # (N, N, 3)
        D = R.norm(dim=-1)            # (N, N)

        # Safe denominator
        D_safe = torch.where(D == 0, torch.ones_like(D), D)

        # Unit vectors
        D3 = D_safe[:, :, None]
        R_hat = R / D3                                       # (N, N, 3)

        # Force magnitudes
        M = torch.outer(self.masses, self.masses)            # (N, N)
        F = self._G * M / D_safe**2                          # (N, N)

        # Zero diagonal, overlapping pairs
        R_sum = self.radii[:, None] + self.radii[None, :]
        
        overlap = (D == 0) | (D < R_sum)
        F = torch.where(overlap, torch.zeros_like(F), F)

        # Force vectors and net force
        F_vec = F[:, :, None] * R_hat                       # (N, N, 3)
        F_net = F_vec.sum(dim=1)                            # (N, 3)

        accs = F_net / self.masses[:, None]

        return accs

    def G(self):
        return self._G


    def gravity_at(self, pos, t):
        pos = pos.to(dtype=torch.float64)
        if self.online:
            raise Exception("This method can only be used in offline mode!")

        #get all positions
        poss = get_interp_positions(self.tensors, t, self.tspan)

        #compute displacements
        displ = poss - pos.unsqueeze(0)
        norms = displ.norm(dim=1)
        norms = norms.unsqueeze(1)
        #normalize (and include distance division directly)
        displ_hat = torch.where(norms == 0, torch.zeros_like(displ), displ / norms**3)


        accs_mag = self._G * self.masses

        accs = accs_mag.unsqueeze(1) * displ_hat

        epsilon = 1e-10
        displ_hat = torch.where(norms < epsilon, torch.zeros_like(displ), displ / norms**3)
        mask = (norms < epsilon)
        accs = torch.where(mask, torch.zeros_like(accs), accs)

        net_acc = accs.sum(0)
        return net_acc

GM_REGISTRY = {
    "quadratic": QuadraticGravityModel
}