import numpy as np

wanted_radius = 0.0001 #in AU

G = 39.478418

M_earth = 0.000003003

v_y_earth = 6.283185

v = np.sqrt((G*M_earth)/wanted_radius)

print(v)

print(v + v_y_earth)