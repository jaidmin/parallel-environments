from multiprocessing import Process


class EmulatorRunner(Process):
    def __init__(self, id, environments, variables, queue, barrier):
        super().__init__()
        self.id = id
        self.environments = environments
        self.variables = variables
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
            for i, (env, action) in enumerate(self.environments, self.variables[-1]):
                # implem detail: is action a one hot, then np.argmax(action) is needed
                next_state, reward, done, _ = env.step(action)
                if done:
                    self.variables[0][i] = env.reset()
                else:
                    self.variables[0][i] = next_state
                self.variables[1][i] = reward
                self.variables[2][i] = done
            count += 1
            self.barrier.put(True)
