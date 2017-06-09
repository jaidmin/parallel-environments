from multiprocessing import Queue, Process
import numpy as np
import multiprocessing, logging

class Worker(Process):
    def __init__(self, id, environments, states, rewards, done, actions, queue, barrier):
        super().__init__()
        self.id = id
        self.environments = environments
        self.states = states
        self.rewards = rewards
        self.done = done
        self.actions = actions
        self.queue = queue
        self.barrier = barrier

    def run(self):
        super().run()
        self._run()

    def _run(self):
        count = 0
        while True:
            instruction = self.queue.get()
            if instruction is None:
                break
            for i in range(len(self.environments)):
                env = self.environments[i]
                action = self.actions[i]

                # implem detail: is action a one hot, then np.argmax(action) is needed
                # sidenote: action should neither be one hot, nor the wrong scale, take care of that in aciton selection
                action = np.argmax(action)
                next_state, reward, done = env.step(action)
                if done:
                    self.states[i] = env.reset()
                else:
                    self.states[i] = next_state
                self.rewards[i] = reward
                self.done[i] = done
            count += 1
            self.barrier.put(True)