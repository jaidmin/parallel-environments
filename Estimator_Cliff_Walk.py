import tensorflow as tf
import numpy as np


class Estimator:
    def __init__(self, num_actions, num_states, num_hidden_units, lr):
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_hidden_units = num_hidden_units

        self.inp_scalar = tf.placeholder(tf.int32, (None,), name="input_scalar")
        self.batch_size = tf.shape(self.inp_scalar)[0]

        self.chosen_actions = tf.placeholder(tf.int32, (None,), name="chosen_action_index")
        self.chosen_actions_indices = tf.concat(
            [tf.expand_dims(tf.range(self.batch_size), -1), tf.expand_dims(self.chosen_actions, -1)], axis=1)

        self.targets = tf.placeholder(tf.float32, (None,), name="target")

        self.inp_one_hot = tf.one_hot(self.inp_scalar, num_states, dtype=tf.float32)
        self.dense_1 = tf.layers.dense(self.inp_one_hot, num_hidden_units, activation=tf.nn.relu, name="dense_1")
        self.out = tf.layers.dense(self.dense_1, num_actions, activation=None, name="out")

        self.affected_out = tf.gather_nd(self.out, self.chosen_actions_indices)

        self.loss = tf.losses.mean_squared_error(self.targets, self.affected_out)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.loss)
        self.session = tf.Session()
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

    def predict_q_values(self, states):
        feed_dict = {"input_scalar:0": states}
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
        feed_dict = {"input_scalar:0": states, "chosen_action_index:0": chosen_action_index, "target:0": target}
        loss, _ = self.session.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss


def get_lr(start, stop, steps, curr_step):
    return start - (curr_step / steps) * (start - stop)


def get_epsilon(start, stop, steps, curr_step):
    return start - (curr_step / steps) * (start - stop)
