import tensorflow as tf
import numpy as np


class Estimator:
    # implemt global step somehow...
    def __init__(self, num_actions, lr):
        self.num_actions = num_actions

        self.input = tf.placeholder(tf.float32, (None,84,84), name="input")
        # think about how to get last 4 frames of history, think that is important
        self.batch_size = tf.shape(self.input)[0]

        self.chosen_actions = tf.placeholder(tf.int32, (None,), name="chosen_action_index")
        self.chosen_actions_indices = tf.concat(
            [tf.expand_dims(tf.range(self.batch_size), -1), tf.expand_dims(self.chosen_actions, -1)], axis=1)

        self.targets = tf.placeholder(tf.float32, (None,), name="target")

        self.conv_1 = tf.layers.conv2d(tf.expand_dims(self.input, -1), 16, (8,8), (4,4), activation=tf.nn.relu, name="conv_1")
        self.conv_2 = tf.layers.conv2d(self.conv_1, 32, (4,4), (2,2), activation=tf.nn.relu, name="conv_2")
        # maybe reshape here in the future
        self.flattened = tf.contrib.layers.flatten(self.conv_2)

        self.dense_1 = tf.layers.dense(self.flattened, 256, activation=tf.nn.relu, name="dense_1")
        self.out = tf.layers.dense(self.dense_1, num_actions, activation=None, name="out")

        self.affected_out = tf.gather_nd(self.out, self.chosen_actions_indices)

        self.loss = tf.losses.mean_squared_error(self.targets, self.affected_out)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.loss)
        self.session = tf.Session()
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

    def predict_q_values(self, states):
        feed_dict = {"input:0": states}
        return self.session.run(self.out, feed_dict=feed_dict)

    def choose_greedy_actions(self, states):
        return np.argmax(self.predict_q_values(states), axis=1)

    def choose_e_greedy_actions(self, states, epsilon):
        # this has to be batched !!! do later # batching done ! remove after test
        e_greedy_mask = np.array([np.random.choice([0, 1], p=[(1 - epsilon), epsilon]) for _ in range(len(states))],
                                 dtype=bool)
        # use sum of e greedy mask to look how many random actions to generate! works because it is 1-0 array
        random_actions = np.array([np.random.choice(range(self.num_actions)) for _ in range(sum(e_greedy_mask))])
        greedy_actions = self.choose_greedy_actions(states)
        e_greedy_actions = greedy_actions
        e_greedy_actions[e_greedy_mask] = random_actions
        return e_greedy_actions

    def update(self, states, chosen_action_index, target):
        feed_dict = {"input:0": states, "chosen_action_index:0": chosen_action_index, "target:0": target}
        loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss


def get_lr(start, stop, steps, curr_step):
    return start - (curr_step / steps) * (start - stop)


def get_epsilon(start, stop, steps, curr_step):
    return start - (curr_step / steps) * (start - stop)
