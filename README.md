# Installation

1. make sure python (version 3.11.12) is installed on your system (verify by running `python --version` in CMD/Terminal)
2. run `pip install requirements.txt` in CMD/Terminal.
3. For good measure, also run `pip install .`
4. If all of these were successful, try running `python test.py`. This should start the simulation, which will first compute planet trajectories and then simulate the mission with the given thrust schedule in the file `test_sched.py` (this one is written by AI, it is quite shitty). You can then adjust everything in `test_sched.py`, you do not need to edit anything anywhere else. What it MUST contain is a function `thrust_fn(t, state)`, that's the minimum requirement. "state" here is just the spacecraft position and velocity in the absolute reference frame, though it shouldn't be necessary. `thrust_fn` needs to return a torch.Tensor which represents the thrust force (3-d), see example code.