"""
Optimized Earth-to-Moon thrust schedule.

Generated from trajectory data covering a 30-day solar system ephemeris.
Optimizer: grid search + Nelder-Mead over (TLI angle, LOI angle, coast multiplier).

Mission profile
---------------
  Phase 1 — Trans-Lunar Injection (TLI)
    t =      0 s  →  159 966 s  (44.44 hr)
    Thrust:  Tx =  5.637 N,  Ty = 19.189 N  (73.63° in Sun-inertial frame)
    Δv:      ≈ 2 000 m/s  (prograde burn)

  Phase 2 — Coast (engines off)
    t = 159 966 s  →  583 499 s  (117.65 hr / 4.90 days)

  Phase 3 — Lunar Orbit Insertion (LOI)
    t = 583 499 s  →  660 903 s  (21.50 hr)
    Thrust:  Tx =  5.661 N,  Ty = −19.182 N  (−73.56° in Sun-inertial frame)
    Δv:      ≈ 1 634 m/s  (retrograde capture burn)

  Total duration: 7.649 days
  Fuel consumed:  1 356 kg / 2 000 kg  (67.8 %)
  Final mass:       743.6 kg  (dry 100 kg + 643.6 kg reserve)
  Moon approach:  1 892 km from surface  (target 1 835 km, Δ ≈ 57 km)
  Relative speed: 1 788 m/s             (circular orbit target 1 634 m/s)

Spacecraft parameters
---------------------
  m_dry  = 100 kg
  m_fuel = 2 000 kg  →  m0 = 2 100 kg
  v_e    = 3 500 m/s  (exhaust velocity)
  T_max  =    20 N

Usage
-----
  from thrust_fn import thrust_fn
  import torch

  F = thrust_fn(t=0.0)           # → tensor([5.637, 19.189, 0.])   (N)
  F = thrust_fn(t=300_000.0)     # → tensor([0., 0., 0.])           (coast)
  F = thrust_fn(t=600_000.0)     # → tensor([5.661, -19.182, 0.])   (N)

  # Vectorised over a time array
  import torch
  times = torch.linspace(0, 660_903, 1000)
  forces = torch.stack([thrust_fn(t.item()) for t in times])  # (1000, 3)
"""

import math
import torch

# ── Optimised parameters (Sun-inertial frame, SI units) ──────────────────────

# Phase boundaries (seconds)
_T_TLI_START: float =       0.0       # TLI ignition
_T_TLI_END:   float = 159_966.090     # TLI cutoff   (44.435 hr)
_T_LOI_START: float = 583_499.582     # LOI ignition (162.08 hr)
_T_LOI_END:   float = 660_903.579     # LOI cutoff / mission end (183.58 hr)

# Thrust vectors (Newtons, 2-D coplanar transfer in xy-plane, z = 0)
_T_MAX: float = 20.0   # N

# TLI: 73.63° in Sun-inertial frame
_TLI_ANGLE: float = 1.28505814   # rad
_TLI_X: float = _T_MAX * math.cos(_TLI_ANGLE)   #  5.637 N
_TLI_Y: float = _T_MAX * math.sin(_TLI_ANGLE)   # 19.189 N

# LOI: −73.56° in Sun-inertial frame (retrograde capture)
_LOI_ANGLE: float = -1.28383216  # rad
_LOI_X: float = _T_MAX * math.cos(_LOI_ANGLE)   #  5.661 N
_LOI_Y: float = _T_MAX * math.sin(_LOI_ANGLE)   # -19.182 N

# Pre-built tensors (avoids repeated allocation in tight loops)
_F_TLI:   torch.Tensor = torch.tensor([_TLI_X, _TLI_Y, 0.0], dtype=torch.float64)
_F_LOI:   torch.Tensor = torch.tensor([_LOI_X, _LOI_Y, 0.0], dtype=torch.float64)
_F_ZERO:  torch.Tensor = torch.zeros(3, dtype=torch.float64)


def thrust_fn(t: float, state) -> torch.Tensor:
    """
    Return the thrust force vector [Fx, Fy, Fz] (Newtons) at time t (seconds).

    The trajectory is coplanar in the xy-plane of the Sun-inertial frame,
    so Fz is always 0.  All values are in SI units.

    Parameters
    ----------
    t : float
        Elapsed mission time in seconds.  t = 0 corresponds to the moment
        the spacecraft departs its initial circular Earth orbit.

    Returns
    -------
    torch.Tensor  shape (3,)  dtype float64
        Force vector [Fx, Fy, Fz] in Newtons.
    """
    if _T_TLI_START <= t < _T_TLI_END:
        return _F_TLI.clone()

    if _T_LOI_START <= t < _T_LOI_END:
        return _F_LOI.clone()

    return _F_ZERO.clone()


# ── Optional: smooth ramp at burn edges (avoids step discontinuity) ──────────

_RAMP_S: float = 60.0   # ramp half-width in seconds (1 min)

def thrust_fn_smooth(t: float, ramp_s: float = _RAMP_S) -> torch.Tensor:
    """
    Like thrust_fn but applies a smooth sigmoid ramp over `ramp_s` seconds
    at each ignition/cutoff edge.  Useful if your integrator is sensitive to
    step discontinuities.

    Parameters
    ----------
    t      : float  — mission time (seconds)
    ramp_s : float  — half-width of the sigmoid ramp (default 60 s)

    Returns
    -------
    torch.Tensor  shape (3,)  dtype float64
    """
    def _sigmoid(x: float) -> float:
        v = -x / ramp_s
        if v > 500:   return 0.0
        if v < -500:  return 1.0
        return 1.0 / (1.0 + math.exp(v))

    # TLI ramp: on at T_TLI_START, off at T_TLI_END
    tli_scale = (_sigmoid(t - _T_TLI_START) *
                 _sigmoid(_T_TLI_END - t))

    # LOI ramp: on at T_LOI_START, off at T_LOI_END
    loi_scale = (_sigmoid(t - _T_LOI_START) *
                 _sigmoid(_T_LOI_END - t))

    fx = tli_scale * _TLI_X + loi_scale * _LOI_X
    fy = tli_scale * _TLI_Y + loi_scale * _LOI_Y
    return torch.tensor([fx, fy, 0.0], dtype=torch.float64)


# ── Convenience: mission phase label ─────────────────────────────────────────

def mission_phase(t: float) -> str:
    """Return a human-readable phase label for time t (seconds)."""
    if t < _T_TLI_START:
        return "pre-launch"
    if t < _T_TLI_END:
        return "TLI"
    if t < _T_LOI_START:
        return "coast"
    if t < _T_LOI_END:
        return "LOI"
    return "post-insertion"


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        (0.0,              "TLI start"),
        (_T_TLI_END / 2,   "TLI mid"),
        (_T_TLI_END - 1,   "TLI last second"),
        (_T_TLI_END,       "coast start"),
        ((_T_TLI_END + _T_LOI_START) / 2, "coast mid"),
        (_T_LOI_START,     "LOI start"),
        (_T_LOI_END - 1,   "LOI last second"),
        (_T_LOI_END,       "post-insertion"),
    ]

    print(f"{'Time (s)':>14}  {'Phase':<18}  {'Fx (N)':>10}  {'Fy (N)':>10}  {'|F| (N)':>9}")
    print("-" * 70)
    for t, label in test_cases:
        F = thrust_fn(t)
        mag = float(F.norm())
        print(f"{t:>14.1f}  {label:<18}  {F[0].item():>10.4f}  {F[1].item():>10.4f}  {mag:>9.4f}")

    print("\nSmooth variant at TLI ignition edge:")
    for t in [_T_TLI_START - 120, _T_TLI_START - 60, _T_TLI_START,
              _T_TLI_START + 60, _T_TLI_START + 120]:
        F = thrust_fn_smooth(t)
        print(f"  t={t:+.0f} s  |F|={float(F.norm()):.4f} N  phase={mission_phase(t)}")