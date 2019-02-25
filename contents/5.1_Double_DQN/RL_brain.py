import tensorflow as tf
import numpy as np

class Double_DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate = 0.01,
            reward_decay = 0.9,
            e_greedy = 0.9,
            replace_target_iter = 300,
            memory_size = 500,
            batch_size = 32,
            e_greedy_increment = None,
            output_graph=False
            ):
        self.n_actions = n_actions
        self.n_features = n_features

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.sess = tf.Session()
        self.learn_step_counter = 0
        self.build_net()

        t_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('hard_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_param, e_param)]

        self.sess.run(tf.global_variables_initializer())
        self.cost_his=[]

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replace\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        # -------------------- add part
        q_next, q_eval4next = self.sess.run([self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation

        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        max_act4next = np.argmax(q_eval4next, axis=1)  # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # --------------------

        _, cost = self.sess.run([self._train_op, self.loss],
                        feed_dict={
                                self.s: batch_memory[:, :self.n_features],
                                self.a: batch_memory[:, self.n_features],      
                                self.s_: batch_memory[:, -self.n_features:],
                                self.q_target: q_target
                        }
                               )

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        trainsition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = trainsition
        self.memory_counter += 1

    def build_net(self):
        # ---------net input
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.a = tf.placeholder(tf.int32, [None, ],  name='a')
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        w_init, b_init = tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)

        e1 = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_init, bias_initializer=b_init, name='e1')
        self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_init, bias_initializer=b_init, name='q')

        t1 = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_init, bias_initializer=b_init, name='t1')
        self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_init, bias_initializer=b_init, name='t2')

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval, name='td_error'))
        self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)

        return  action

