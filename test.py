from simply.bodies.spacecraft import SpaceCraftBody
from simply import animate_interactive
from simply.universe import Universe

uni = Universe("universe.yaml")

uni.simulate(0, 0.2, 0.01)

sc = SpaceCraftBody("spacecraft.yaml")


sc_states = sc.simulate(0, 0.2, 0.00001)

animate_interactive("traj.pt", sc_states=sc_states)