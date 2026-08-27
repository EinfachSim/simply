import bisect

#computes hermite interpolation at t
def hermite_eval(t, t0, t1, s0, s1):

    p0s, v0s = s0[:, :3], s0[:, 3:]
    p1s, v1s = s1[:, :3], s1[:, 3:]

    tau = (t - t0) / (t1 - t0)
    dt = t1 - t0

    h00 = 2 * tau**3 - 3*tau**2 + 1
    h10 = tau**3 - 2*tau**2 + tau
    h01 = -2*tau**3 + 3*tau**2
    h11 = tau**3 - tau**2

    return h00*p0s + h10*v0s*dt + h01*p1s + h11*v1s*dt

def get_interp_positions(states, t, times):
        
        i = bisect.bisect_right(times, t) - 1
        i = min(i, len(times) - 2)

        t0, t1 = times[i], times[i+1]
        state0 = states[i]
        state1 = states[i+1]

        return hermite_eval(t, t0, t1, state0, state1)