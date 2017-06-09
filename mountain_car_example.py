import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from ParallelEnvironment import EnvironmentProvider, ParallelEnvironment

class Estimator():
    """
    Value Function approximator. 
    """

    def __init__(self):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(3):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    states, rewards, done, actions = env.get_shared_variables()
    env.start()




    for j in range(num_episodes):
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** j, 3)


        for i in range(env.num_envs):
            action_probs = policy(states[i])
            action_ind = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action = np.eye(3)[action_ind]
            actions[i] = action

        old_states = np.copy(states)

        env.update_environments()
        env.wait_updated()



        for k in range(env.num_envs):
            q_values_next = estimator.predict(states[k])
            td_target = rewards[k] + discount_factor * np.max(q_values_next)
            estimator.update(old_states[k], np.argmax(actions[k]), td_target)

        print("currently in episode: {}".format(j))
    env.stop()

    return estimator






def play_game(estimator):
    env = gym.make("MountainCar-v0")

    state = env.reset()
    policy = make_epsilon_greedy_policy(estimator, 0, 3)

    action = policy(state)
    action = np.argmax(action)

    while True:
        done = False


        while not done:
            env.render()
            state, reward, done, info = env.step(action)
            action = policy(state)
            action = np.argmax(action)




if __name__ == "__main__":

    env = gym.envs.make("MountainCar-v0")

    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
            ])
    featurizer.fit(scaler.transform(observation_examples))

    estimator = Estimator()

    provider = EnvironmentProvider("MountainCar-v0",False)
    env_parallel = ParallelEnvironment(provider,8,4,3)


    estimator = q_learning(env_parallel, estimator, 20000, epsilon=0.0)

    play_game(estimator)

