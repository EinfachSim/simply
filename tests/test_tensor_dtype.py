"""
test_dtype.py

Checks that every floating-point tensor touched during a simulation
is float64. Uses torch's dispatch mechanism to intercept all tensor
operations at runtime — catches both construction-time and
computation-time float32 tensors.

Run with:
    pytest tests/test_dtype.py -v
"""

import pytest
import torch
from torch.overrides import TorchFunctionMode

from simply.universe import Universe
from simply.bodies.spacecraft import SpaceCraftBody


# ------------------------------------------------------------------ #
#  Dispatch hook: intercepts every torch operation                    #
# ------------------------------------------------------------------ #

class Float32Detector(TorchFunctionMode):
    """
    A TorchFunctionMode that records every floating-point tensor
    that is NOT float64. Attach via `with Float32Detector() as d:`.
    """

    def __init__(self):
        self.violations: list[dict] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        result = func(*args, **kwargs)

        # Check all tensor arguments and the result
        for t in self._iter_tensors(args) + self._iter_tensors(result):
            if t.is_floating_point() and t.dtype != torch.float64:
                self.violations.append({
                    "op":    func.__name__ if hasattr(func, "__name__") else str(func),
                    "dtype": t.dtype,
                    "shape": tuple(t.shape),
                })

        return result

    @staticmethod
    def _iter_tensors(x) -> list[torch.Tensor]:
        """Recursively collect tensors from args/kwargs/results."""
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, (list, tuple)):
            out = []
            for item in x:
                out.extend(Float32Detector._iter_tensors(item))
            return out
        return []


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _format_violations(violations: list[dict]) -> str:
    lines = [f"  [{i+1}] op={v['op']}  dtype={v['dtype']}  shape={v['shape']}"
             for i, v in enumerate(violations)]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #

class TestFloat64Universe:

    def test_universe_state_dtype(self):
        """System state tensor must be float64 after construction."""
        uni = Universe("universe.yaml")
        assert uni.system_state.dtype == torch.float64, (
            f"system_state is {uni.system_state.dtype}, expected float64"
        )

    def test_gravity_model_tensors_dtype(self):
        """Masses and radii stored in the gravity model must be float64."""
        uni = Universe("universe.yaml")
        gm = uni.gravity_model
        assert gm.masses.dtype == torch.float64, (
            f"gravity_model.masses is {gm.masses.dtype}"
        )
        assert gm.radii.dtype == torch.float64, (
            f"gravity_model.radii is {gm.radii.dtype}"
        )

    def test_no_float32_during_simulation(self):
        """
        No float32 tensor should be created or computed during Universe.simulate.
        This catches dtype regressions introduced inside any op, not just
        at construction time.
        """
        uni = Universe("universe.yaml")

        with Float32Detector() as detector:
            uni.simulate(0, 0.02, 0.01)

        assert not detector.violations, (
            f"Found {len(detector.violations)} float32 tensor(s) during "
            f"Universe.simulate:\n{_format_violations(detector.violations)}"
        )


class TestFloat64Spacecraft:

    def test_spacecraft_state_dtype(self):
        """Spacecraft initial state must be float64."""
        sc = SpaceCraftBody("spacecraft.yaml")
        assert sc.state().dtype == torch.float64, (
            f"spacecraft state is {sc.state().dtype}, expected float64"
        )

    def test_gravity_model_offline_tensors_dtype(self):
        """Tensors loaded from traj.pt into the offline gravity model must be float64."""
        sc = SpaceCraftBody("spacecraft.yaml")
        gm = sc.gravity_model
        assert gm.tensors.dtype == torch.float64, (
            f"offline gravity tensors are {gm.tensors.dtype}"
        )
        assert gm.masses.dtype == torch.float64, (
            f"offline gravity masses are {gm.masses.dtype}"
        )

    def test_no_float32_during_spacecraft_simulation(self):
        """
        No float32 tensor should appear during SpaceCraftBody.simulate.
        Requires a traj.pt to already exist (run Universe.simulate first).
        """
        sc = SpaceCraftBody("spacecraft.yaml")

        with Float32Detector() as detector:
            sc.simulate(0, 0.02, 0.00001)

        assert not detector.violations, (
            f"Found {len(detector.violations)} float32 tensor(s) during "
            f"SpaceCraftBody.simulate:\n{_format_violations(detector.violations)}"
        )


class TestFloat64GravityModel:

    def test_acceleration_output_dtype(self):
        """gravity_model() output must be float64."""
        uni = Universe("universe.yaml")
        poss = uni.system_state[:, :3]
        accs = uni.gravity_model(poss)
        assert accs.dtype == torch.float64, (
            f"acceleration output is {accs.dtype}"
        )

    def test_gravity_at_output_dtype(self):
        """gravity_at() output must be float64 (offline mode)."""
        # Needs traj.pt — run universe sim first if missing
        import os
        if not os.path.exists("traj.pt"):
            uni = Universe("universe.yaml")
            uni.simulate(0, 0.02, 0.01)

        sc = SpaceCraftBody("spacecraft.yaml")
        pos = sc.state()[:3]
        acc = sc.gravity_model.gravity_at(pos, t=0.01)
        assert acc.dtype == torch.float64, (
            f"gravity_at output is {acc.dtype}"
        )