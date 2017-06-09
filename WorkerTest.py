import multiprocessing
import numpy as np
from ParallelEnvironment import *
import time
if __name__ == "__main__":
    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8:c_uint}

    provider = EnvironmentProvider("Pong-v0")
    queue = multiprocessing.Queue()
    queue.put(True)
    barrier = multiprocessing.Queue()
    envs = [provider.create() for _ in range(4)]

    for env in envs:
        env.reset()


    def _get_shared(array):
        dtype = NUMPY_TO_C_DTYPE[array.dtype.type]
        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    array = np.array([0,0,0,0], dtype=np.float32)
    states_arr = np.array([env.reset()]* 4)

    a = _get_shared(states_arr)
    b = _get_shared(array)
    c = _get_shared(array)
    d = _get_shared(array)


    worker = Worker(0, envs, a,b,c,d,queue, barrier )

    worker.start()


    print("hello")

    worker.queue.put(True)

    print("worker should execute something")

    worker.queue.put(True)

    print("worker should execute something 2 ")


    worker.queue.put(True)



    print("worker should execute something 2 ")


    worker.queue.put(True)

    time.sleep(5)
    print(worker.states)
