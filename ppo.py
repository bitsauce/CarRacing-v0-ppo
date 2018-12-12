import tensorflow as tf
import numpy as np
import re
import os
import shutil

class PolicyGraph():
    def __init__(self, input_states, taken_actions, num_actions, action_min, action_max, scope_name,
                 initial_mean_factor=0.1, clip_action_space=False):
        with tf.variable_scope(scope_name):
            # Construct model
            self.conv1           = tf.layers.conv2d(input_states, filters=16, kernel_size=8, strides=4, activation=tf.nn.leaky_relu, padding="valid", name="conv1")
            self.conv2           = tf.layers.conv2d(self.conv1, filters=32, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, padding="valid", name="conv2")
            self.shared_features = tf.layers.flatten(self.conv2, name="flatten")
            
            # Policy branch π(a_t | s_t; θ)
            self.action_mean = tf.layers.dense(self.shared_features, num_actions,
                                               activation=tf.nn.tanh,
                                               kernel_initializer=tf.initializers.variance_scaling(scale=initial_mean_factor),
                                               name="action_mean")
            self.action_mean = action_min + ((self.action_mean + 1) / 2) * (action_max - action_min)
            self.action_logstd = tf.get_variable("action_logstd", [num_actions], initializer=tf.zeros_initializer())

            # Value branch V(s_t; θ)
            self.value = tf.layers.dense(self.shared_features, 1, activation=None, name="value")
        
            # Create graph for sampling actions
            self.action_normal  = tf.distributions.Normal(self.action_mean, tf.exp(self.action_logstd), validate_args=True)
            self.sampled_action = tf.squeeze(self.action_normal.sample(1), axis=0)
            if clip_action_space:
                num_envs   = tf.shape(self.sampled_action)[0]
                action_min = tf.reshape(tf.tile(tf.convert_to_tensor(action_min, dtype=tf.float32), (num_envs,)), (num_envs, num_actions))
                action_max = tf.reshape(tf.tile(tf.convert_to_tensor(action_max, dtype=tf.float32), (num_envs,)), (num_envs, num_actions))
                self.sampled_action = tf.clip_by_value(self.sampled_action, action_min, action_max)
            
            # Get the log probability of taken actions
            # log π(a_t | s_t; θ)
            self.action_log_prob = tf.reduce_sum(self.action_normal.log_prob(taken_actions), axis=-1, keepdims=True)
            
            # Validate values
            self.action_mean     = tf.check_numerics(self.action_mean,     "Invalid value for self.action_mean")
            self.action_logstd   = tf.check_numerics(self.action_logstd,   "Invalid value for self.action_logstd")
            self.value           = tf.check_numerics(self.value,           "Invalid value for self.value")
            self.action_log_prob = tf.check_numerics(self.action_log_prob, "Invalid value for self.action_log_prob")

class PPO():
    def __init__(self, num_actions, input_shape, action_min, action_max,
                 epsilon=0.2, value_scale=0.5, entropy_scale=0.01,
                 model_checkpoint=None, model_name="ppo"):
        tf.reset_default_graph()
        
        self.input_states  = tf.placeholder(shape=(None, *input_shape), dtype=tf.float32, name="input_state_placeholder")
        self.taken_actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name="taken_action_placeholder")
        self.input_states  = tf.check_numerics(self.input_states, "Invalid value for self.input_states")
        self.taken_actions = tf.check_numerics(self.taken_actions, "Invalid value for self.taken_actions")
        self.policy        = PolicyGraph(self.input_states, self.taken_actions, num_actions, action_min, action_max, "policy")
        self.policy_old    = PolicyGraph(self.input_states, self.taken_actions, num_actions, action_min, action_max, "policy_old")

        # Create policy gradient train function
        self.returns   = tf.placeholder(shape=(None,), dtype=tf.float32, name="returns_placeholder")
        self.advantage = tf.placeholder(shape=(None,), dtype=tf.float32, name="advantage_placeholder")
        
        # Calculate ratio:
        # r_t(θ) = exp( log   π(a_t | s_t; θ) - log π(a_t | s_t; θ_old)   )
        # r_t(θ) = exp( log ( π(a_t | s_t; θ) /     π(a_t | s_t; θ_old) ) )
        # r_t(θ) = π(a_t | s_t; θ) / π(a_t | s_t; θ_old)
        self.prob_ratio = tf.exp(self.policy.action_log_prob - self.policy_old.action_log_prob)
        
        # Validate values
        self.returns = tf.check_numerics(self.returns, "Invalid value for self.returns")
        self.advantage = tf.check_numerics(self.advantage, "Invalid value for self.advantage")
        self.prob_ratio = tf.check_numerics(self.prob_ratio, "Invalid value for self.prob_ratio")

        # Policy loss
        adv = tf.expand_dims(self.advantage, axis=-1)
        self.policy_loss = tf.reduce_mean(tf.minimum(self.prob_ratio * adv, tf.clip_by_value(self.prob_ratio, 1.0 - epsilon, 1.0 + epsilon) * adv))

        # Value loss = mse(V(s_t) - R_t)
        self.value_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.policy.value), self.returns)) * value_scale
        
        # Entropy loss
        self.entropy_loss = tf.reduce_mean(tf.reduce_sum(self.policy.action_normal.entropy(), axis=-1)) * entropy_scale
        
        # Total loss
        self.loss = -self.policy_loss + self.value_loss - self.entropy_loss
        
        # Policy parameters
        policy_params     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy/")
        policy_old_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_old/")
        assert(len(policy_params) == len(policy_old_params))
        for src, dst in zip(policy_params, policy_old_params):
            assert(src.shape == dst.shape)

        # Minimize loss
        self.learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name="lr_placeholder")
        self.optimizer     = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step    = self.optimizer.minimize(self.loss, var_list=policy_params)

        # Update network parameters
        self.update_op = tf.group([dst.assign(src) for src, dst in zip(policy_params, policy_old_params)])

        # Create session
        self.sess = tf.Session()

        # Run the initializer
        self.sess.run(tf.global_variables_initializer())
        
        # Summaries
        tf.summary.scalar("loss_policy", self.policy_loss)
        tf.summary.scalar("loss_value", self.value_loss)
        tf.summary.scalar("loss_entropy", self.entropy_loss)
        tf.summary.scalar("loss", self.loss)
        for i in range(num_actions):
            tf.summary.scalar("taken_actions_{}".format(i), tf.reduce_mean(self.taken_actions[:, i]))
            tf.summary.scalar("prob_ratio_{}".format(i), tf.reduce_mean(self.prob_ratio[i]))
            tf.summary.scalar("policy.action_mean_{}".format(i), tf.reduce_mean(self.policy.action_mean[:, i]))
            tf.summary.scalar("policy.action_std_{}".format(i), tf.reduce_mean(tf.exp(self.policy.action_logstd[i])))
        tf.summary.scalar("returns", tf.reduce_mean(self.returns))
        tf.summary.scalar("advantage", tf.reduce_mean(self.advantage))
        tf.summary.scalar("learning_rate", tf.reduce_mean(self.learning_rate))
        self.summary_merged = tf.summary.merge_all()
        
        # Load model checkpoint
        self.model_name = model_name
        self.saver = tf.train.Saver()

        self.model_dir = "./models/{}".format(self.model_name)
        self.log_dir   = "./logs/{}".format(self.model_name)
        self.video_dir = "./videos/{}".format(self.model_name)
        if model_checkpoint is None and os.path.isdir(self.model_dir):
            answer = input("{} exists. Do you wish to resume training? (Y/N) ".format(self.model_dir))
            if answer.upper() == "Y":
                model_checkpoint = tf.train.latest_checkpoint(self.model_dir)
            else:
                answer = input("Do you wish delete and restart training? (Y/N) ")
                if answer.upper() != "Y":
                    raise Exception("Trained model exists in destination {}." +
                                    "Please delete it or change model_name and try again".format(self.model_dir))

        if model_checkpoint:
            self.step_idx = int(re.findall(r"[/\\]step\d+", model_checkpoint)[0][len("/step"):])
            self.saver.restore(self.sess, model_checkpoint)
            print("Model checkpoint restored from {}".format(model_checkpoint))
        else:
            self.step_idx = 0
            for d in [self.model_dir, self.log_dir, self.video_dir]:
                if os.path.isdir(d): shutil.rmtree(d)
                os.makedirs(d)

        self.train_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
    def save(self):
        model_checkpoint = os.path.join(self.model_dir, "step{}.ckpt".format(self.step_idx))
        self.saver.save(self.sess, model_checkpoint)
        print("Model checkpoint saved to {}".format(model_checkpoint))
        
    def train(self, input_states, taken_actions, returns, advantage, learning_rate=1e-4):
        r = self.sess.run([self.summary_merged, self.train_step, self.loss, self.policy_loss, self.value_loss, self.entropy_loss],
                          feed_dict={self.input_states: input_states,
                                     self.taken_actions: taken_actions,
                                     self.returns: returns,
                                     self.advantage: advantage,
                                     self.learning_rate: learning_rate(self.step_idx) if callable(learning_rate) else learning_rate})
        self.train_writer.add_summary(r[0], self.step_idx)
        self.step_idx += 1
        return r[2:]
        
    def predict(self, input_states, use_old_policy=False, greedy=False):
        policy = self.policy_old if use_old_policy else self.policy
        action = policy.action_mean if greedy else policy.sampled_action
        return self.sess.run([action, policy.value],
                             feed_dict={self.input_states: input_states})

    def write_to_summary(self, name, value):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        self.train_writer.add_summary(summary, self.step_idx)

    def update_old_policy(self):
        self.sess.run(self.update_op)
