import multiprocessing
import numpy as np
from ParallelEnvironment import *
import time
if __name__ == "__main__":
    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8:c_uint}

    provider = EnvironmentProvider("Pong-v0", False)

    parallel = ParallelEnvironment(provider, 4, 2, 2)

    states, rewards, done, actions = parallel.get_shared_variables()


    while True:
        not_shared_actions = np.zeros((parallel.num_envs, parallel.num_actions))
        for i in range(parallel.num_envs):
            action_ind = np.random.choice([0,1])
            action = np.eye(2)[action_ind]
            not_shared_actions[i] = action

        parallel.step(not_shared_actions)
        print("step taken")





