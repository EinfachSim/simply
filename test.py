from simply.bodies.spacecraft import SpaceCraftBody
from simply import animate_interactive
from simply.universe import Universe

uni = Universe("universe.yaml")

uni.simulate(0, 1, 0.01)

sc = SpaceCraftBody("spacecraft.yaml")

sc_states = sc.simulate(0, 1, 0.00001)

#print(sc_states.shape)


animate_interactive("traj.pt", sc_states=sc_states)