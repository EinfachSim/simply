from .verlet import VerletIntegrator
from .rk4 import RK4Integrator

INTEG_REGISTRY = {
    "verlet": VerletIntegrator,
    "rk4": RK4Integrator
}