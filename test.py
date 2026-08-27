from simply.bodies.spacecraft import SpaceCraftBody
from simply import animate_interactive
from simply.universe import Universe


import torch
def thrust_fn(t, state):
    vel = state[:, 3:6].squeeze(0)
    prograde = vel / vel.norm()
    return 10 * prograde

uni = Universe("universe.yaml")

uni.simulate(0, 3*86_400, 86_400)

sc = SpaceCraftBody("spacecraft.yaml")


sc_states = sc.simulate(0, 3*86_400, 60, thrust_fn=thrust_fn)

animate_interactive("traj.pt", sc_states=sc_states)