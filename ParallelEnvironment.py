import gym
from multiprocessing import Queue
from Environment import Environment
import numpy as np
from multiprocessing.sharedctypes import RawArray
from ctypes import c_float, c_double, c_uint
from Worker import Worker


class EnvironmentProvider:
    """oop construct to create environments, useful for example to give ids to all envs"""

    def __init__(self, env_name, preproc, legal_action_set, rescale_reward, reward_max, reward_min):
        self.env_name = env_name
        self.counter = 0
        self.preproc = preproc
        self.legal_action_set = legal_action_set
        self.reward_max = reward_max
        self.reward_min = reward_min
        self.rescale_reward = rescale_reward

    def create(self):
        env_gym = gym.make(self.env_name)
        env = Environment(env_gym, self.counter, preproc=self.preproc, legal_action_set=self.legal_action_set,
                          do_rescale_reward=self.rescale_reward, reward_max=self.reward_max,
                          rewards_min=self.reward_min)
        self.counter += 1
        return env


class ParallelEnvironmentSerial:
    """legacy implementation of multiple envs in one process"""

    def __init__(self, env_provider, num_envs):
        self.envs = [env_provider.create() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.states = [0] * num_envs
        self.done = [0] * num_envs
        self.rewards = [0] * num_envs

    def reset(self):
        for i in range(self.num_envs):
            self.states[i] = self.envs[i].reset()
        return self.states

    def step(self, actions):
        for i in range(self.num_envs):
            self.states[i], self.rewards[i], self.done[i], _ = self.envs[i].step(actions[i])
        return self.states, self.rewards, self.done


class ParallelEnvironment:
    """Parallelized environment implementation. heavily influenced/based on the runners class from paac
       nevertheless changes are existent. spawns num_workers process on creation, are maintained until stop is called
       """
    NUMPY_TO_C_DTYPE = {np.float32: c_float, np.float64: c_double, np.uint8: c_uint}

    def __init__(self, env_provider, num_envs, num_workers, num_actions):
        self.envs = np.array([env_provider.create() for _ in range(num_envs)])
        self.num_envs = num_envs
        self.num_workers = num_workers
        self.queues = [Queue() for _ in range(num_envs)]
        self.barrier = Queue()
        self.num_actions = num_actions

        self.states = np.asarray([emulator.reset() for emulator in self.envs], dtype=np.uint8)
        self.states = self._get_shared(self.states)
        self.rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.rewards = self._get_shared(self.rewards)
        self.done = np.asarray([False] * self.num_envs, dtype=np.float32)
        self.done = self._get_shared(self.done)

        self.actions = np.zeros((self.num_envs, self.num_actions), dtype=np.float32)
        self.actions = self._get_shared(self.actions)
        # actions are one-hot


        envs_for_worker = np.split(self.envs, self.num_workers)
        states_for_worker = np.split(self.states, self.num_workers)
        rewards_for_worker = np.split(self.rewards, self.num_workers)
        done_for_worker = np.split(self.done, self.num_workers)
        actions_for_worker = np.split(self.actions, self.num_workers)

        self.workers = []
        for i in range(num_workers):
            id = i
            envs = envs_for_worker[i]
            states = states_for_worker[i]
            rewards = rewards_for_worker[i]
            done = done_for_worker[i]
            actions = actions_for_worker[i]
            worker = Worker(id, envs, states, rewards, done, actions, self.queues[i], self.barrier)
            self.workers.append(worker)

        self.start()

    def _get_shared(self, array):
        dtype = self.NUMPY_TO_C_DTYPE[array.dtype.type]
        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def start(self):
        for r in self.workers:
            r.start()

    def stop(self):
        for queue in self.queues:
            queue.put(None)

    def get_shared_variables(self):
        return self.states, self.rewards, self.done, self.actions

    def update_environments(self):
        for queue in self.queues:
            queue.put(True)

    def wait_updated(self):
        for wd in range(self.num_workers):
            self.barrier.get()

    def step(self, actions):
        """not clear what to do with this... i want to follow the gym api but this feels kinda wrong to pass the actions 
        explicitly if i could just modify the shared ones... decide what to do at a later point"""
        for i in range(len((actions))):
            self.actions[i] = np.eye(self.num_actions)[actions[i]]
        self.update_environments()
        self.wait_updated()
        return self.states, self.rewards, self.done

    def reset(self):
        for i in range(len(self.states)):
            self.states[i] = self.envs[i].reset()
        return self.states
