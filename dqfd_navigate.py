import minerl
import gym

import random
import time
import numpy as np
from collections import deque
import os
from os import listdir
from os.path import isfile, join, isdir

import tensorflow as tf
import per_replay as replay


def parse_demo(env_name, rep_buffer, data_path, nsteps=10):
    data = minerl.data.make(env_name, data_dir=data_path)

    demo_num = 0
    # Original implementation used sarsd_iter, which returned only one sample instead of a batch.
    for state, action, reward, next_state, done in data.batch_iter(
            batch_size=1, num_epochs=500, seq_len=2000):
        demo_num += 1

        if demo_num == 50:
            break

        parse_ts = 0
        episode_start_ts = 0
        nstep_gamma = 0.99
        nstep_state_deque = deque()
        nstep_action_deque = deque()
        nstep_rew_list = []
        nstep_nexts_deque = deque()
        nstep_done_deque = deque()
        total_rew = 0.

        batch_size, length = (state['pov'].shape)[:2]
        for i in range(batch_size):
            for j in range(length):
                #action_index = 0

                camera_thresholds = (abs(action['camera'][i][j][0]) +
                                     abs(action['camera'][i][j][1])) / 2.0
                if (camera_thresholds > 2.5):
                    if ((action['camera'][i][j][1] < 0) &
                        (abs(action['camera'][i][j][0]) < abs(
                            action['camera'][i][j][1]))):
                        if (action['attack'][i][j] == 0):
                            action_index = 0
                        else:
                            action_index = 1
                    elif ((action['camera'][i][j][1] > 0) &
                          (abs(action['camera'][i][j][0]) < abs(
                              action['camera'][i][j][1]))):
                        if (action['attack'][i][j] == 0):
                            action_index = 2
                        else:
                            action_index = 3
                    elif ((action['camera'][i][j][0] < 0) &
                          (abs(action['camera'][i][j][0]) > abs(
                              action['camera'][i][j][1]))):
                        if (action['attack'][i][j] == 0):
                            action_index = 4
                        else:
                            action_index = 5
                    elif ((action['camera'][i][j][0] > 0) &
                          (abs(action['camera'][i][j][0]) > abs(
                              action['camera'][i][j][1]))):
                        if (action['attack'][i][j] == 0):
                            action_index = 6
                        else:
                            action_index = 7
                elif (action['forward'][i][j] == 1):
                    if (action['attack'][i][j] == 0):
                        action_index = 8
                    else:
                        action_index = 9
                elif (action['jump'][i][j] == 1):
                    if (action['attack'][i][j] == 0):
                        action_index = 10
                    else:
                        action_index = 11
                else:
                    if (action['attack'][i][j] == 0):
                        continue
                    else:
                        action_index = 12

                game_a = action_index
                curr_obs = state['pov'][i][j]

                _obs = next_state['pov'][i][j]
                _rew = reward[i][j]
                _done = done[i][j].astype(int)

                episode_start_ts += 1
                parse_ts += 1

                _rew = np.sign(_rew) * np.log(1. + np.abs(_rew))

                nstep_state_deque.append(curr_obs)
                nstep_action_deque.append(game_a)
                nstep_rew_list.append(_rew)
                nstep_nexts_deque.append(_obs)
                nstep_done_deque.append(_done)

                if episode_start_ts > 10:
                    add_transition(rep_buffer, nstep_state_deque,
                                   nstep_action_deque, nstep_rew_list,
                                   nstep_nexts_deque, nstep_done_deque, _obs,
                                   False, nsteps, nstep_gamma)

                # if episode done we reset
                if _done:
                    #emptying the deques
                    add_transition(rep_buffer, nstep_state_deque,
                                   nstep_action_deque, nstep_rew_list,
                                   nstep_nexts_deque, nstep_done_deque, _obs,
                                   True, nsteps, nstep_gamma)

                    nstep_state_deque.clear()
                    nstep_action_deque.clear()
                    nstep_rew_list.clear()
                    nstep_nexts_deque.clear()
                    nstep_done_deque.clear()

                    episode_start_ts = 0

                    break

        # replay is over emptying the deques
        add_transition(rep_buffer, nstep_state_deque, nstep_action_deque,
                       nstep_rew_list, nstep_nexts_deque, nstep_done_deque,
                       _obs, True, nsteps, nstep_gamma)
        print('Parse finished. {} expert samples added.'.format(parse_ts))

    return rep_buffer


#handles transitions to add to replay buffer and expert buffer
#next step reward (ns_rew) is a list, the rest are deques
def add_transition(rep_buffer,
                   ns_state,
                   ns_action,
                   ns_rew,
                   ns_nexts,
                   ns_done,
                   current_state,
                   empty_deque=False,
                   ns=10,
                   ns_gamma=0.99,
                   is_done=True):
    ns_rew_sum = 0.
    trans = {}
    if empty_deque:
        # emptying the deques
        while len(ns_rew) > 0:
            for j in range(len(ns_rew)):
                ns_rew_sum += ns_rew[j] * ns_gamma**j

            # state,action,reward,
            # next_state,done, n_step_rew_sum, n_steps later
            # don't use done value because at this point the episode is done
            trans['sample'] = [
                ns_state.popleft(),
                ns_action.popleft(),
                ns_rew.pop(0),
                ns_nexts.popleft(), is_done, ns_rew_sum, current_state
            ]

            rep_buffer.add_sample(trans)
    else:
        for j in range(ns):
            ns_rew_sum += ns_rew[j] * ns_gamma**j

        # state,action,reward,
        # next_state,done, n_step_rew_sum, n_steps later
        trans['sample'] = [
            ns_state.popleft(),
            ns_action.popleft(),
            ns_rew.pop(0),
            ns_nexts.popleft(),
            ns_done.popleft(), ns_rew_sum, current_state
        ]

        rep_buffer.add_sample(trans)


class Qnetwork():
    def __init__(self):
        action_len = 13

        def image_scale(image):
            scale_img = image / 255.0
            return scale_img

        tf.compat.v1.disable_eager_execution()
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.input_img_dq = tf.compat.v1.placeholder(shape=[None, 64, 64, 3],
                                                     dtype=tf.float32)
        self.scale_img_dq = tf.map_fn(image_scale,
                                      self.input_img_dq,
                                      dtype=tf.float32)

        self.input_img_nstep = tf.compat.v1.placeholder(
            shape=[None, 64, 64, 3], dtype=tf.float32)
        self.scale_img_nstep = tf.map_fn(image_scale,
                                         self.input_img_nstep,
                                         dtype=tf.float32)

        self.input_expert_action = tf.compat.v1.placeholder(shape=(None, 2),
                                                            dtype=tf.int32)
        self.input_is_expert = tf.compat.v1.placeholder(shape=[None, 1],
                                                        dtype=tf.float32)
        self.input_expert_margin = tf.compat.v1.placeholder(
            shape=[None, action_len], dtype=tf.float32)

        self.conv1_dq = tf.compat.v1.layers.conv2d(
            inputs=self.scale_img_dq,
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding='VALID',
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal")),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)),
            bias_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)))
        self.conv2_dq = tf.compat.v1.layers.conv2d(
            inputs=self.conv1_dq,
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding='VALID',
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal")),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)),
            bias_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)))
        self.conv3_dq = tf.compat.v1.layers.conv2d(
            inputs=self.conv2_dq,
            filters=64,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='VALID',
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal")),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)),
            bias_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)))
        self.x_dq = tf.compat.v1.layers.flatten(self.conv3_dq)
        self.dq_output = tf.compat.v1.layers.dense(self.x_dq,
                                                   action_len,
                                                   activation=tf.nn.relu)

        self.conv1_nstep = tf.compat.v1.layers.conv2d(
            inputs=self.scale_img_nstep,
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding='VALID',
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal")),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)),
            bias_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)))
        self.conv2_nstep = tf.compat.v1.layers.conv2d(
            inputs=self.conv1_nstep,
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding='VALID',
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal")),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)),
            bias_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)))
        self.conv3_nstep = tf.compat.v1.layers.conv2d(
            inputs=self.conv2_nstep,
            filters=64,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='VALID',
            activation=tf.nn.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1.0,
                mode="fan_avg",
                distribution=("uniform" if False else "truncated_normal")),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)),
            bias_regularizer=tf.keras.regularizers.l2(l=0.5 * (10e-5)))
        self.x_nstep = tf.compat.v1.layers.flatten(self.conv3_nstep)
        self.nstep_output = tf.compat.v1.layers.dense(self.x_nstep,
                                                      action_len,
                                                      activation=tf.nn.relu)

        self.targetQ_dq = tf.compat.v1.placeholder(shape=[None],
                                                   dtype=tf.float32)
        self.targetQ_nstep = tf.compat.v1.placeholder(shape=[None],
                                                      dtype=tf.float32)

        self.actions = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,
                                         action_len,
                                         dtype=tf.float32)

        # Supervised Large Margin Classification
        elems = [
            self.dq_output, self.actions, self.input_is_expert,
            self.input_expert_margin
        ]

        def slmc_operator(slmc_input):
            is_exp = tf.cast(slmc_input[2], dtype=tf.float32)
            sa_values = slmc_input[0]
            exp_act = tf.cast(slmc_input[1], dtype=tf.int32)
            exp_margin = slmc_input[3]

            exp_val = tf.gather(sa_values, exp_act)
            max_margin_1 = tf.reduce_max(input_tensor=sa_values + exp_margin)
            max_margin_2 = max_margin_1 - exp_val
            max_margin_3 = tf.multiply(is_exp, max_margin_2)

            return max_margin_3

        self.slmc_output = tf.map_fn(slmc_operator, elems, dtype=tf.float32)

        self.Q_dq = tf.reduce_sum(input_tensor=tf.multiply(
            self.dq_output, self.actions_onehot),
                                  axis=1)
        self.Q_nstep = tf.reduce_sum(input_tensor=tf.multiply(
            self.nstep_output, self.actions_onehot),
                                     axis=1)

        self.slmc = tf.reduce_mean(input_tensor=tf.abs(self.slmc_output))
        self.td_error_dq = tf.reduce_mean(
            input_tensor=tf.square(self.targetQ_dq - self.Q_dq))
        self.td_error_nstep = tf.reduce_mean(
            input_tensor=tf.square(self.targetQ_nstep - self.Q_nstep))

        self.loss = tf.reduce_sum(input_tensor=self.td_error_dq +
                                  self.td_error_nstep + self.slmc)

        self.summaries = tf.compat.v1.summary.merge(
            [tf.compat.v1.summary.scalar("loss_VALUE", self.loss)])

        self.trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


def main():
    env_name = "MineRLNavigate-v0"
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        root_path, 'data')  # Or wherever you have stored the expert data

    # 1) Get Expert Data
    expert_buffer = replay.PrioritizedReplayBuffer(75000,
                                                   alpha=0.4,
                                                   beta=0.6,
                                                   epsilon=0.001)
    expert_buffer = parse_demo(env_name, expert_buffer, data_path)

    # 2) Train Expert Model on Data
    model = Qnetwork()

    saver = tf.compat.v1.train.Saver()
    trainables = tf.compat.v1.trainable_variables()
    init = tf.compat.v1.global_variables_initializer()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(init)

    summary_path = root_path + 'train_summary/' + env_name
    summary_writer = tf.compat.v1.summary.FileWriter(summary_path)

    dqfd_model_path = root_path + 'dqfd_model'
    expert_model_path = root_path + 'expert_model'

    action_len = 13
    train_steps = 100000
    batch_size = 32
    gamma = 0.99
    nstep_gamma = 0.99
    exp_margin_constant = 0.8

    time_int = int(time.time())
    loss = np.zeros((4, ))

    print('Training expert model')
    for current_step in range(train_steps):
        print("current_step: " + str(current_step))

        empty_batch_by_one = np.zeros((batch_size, 1))
        empty_action_batch = np.zeros((batch_size, 2))
        empty_action_batch[:, 0] = np.arange(batch_size)
        empty_batch_by_action_len = np.zeros((batch_size, action_len))
        ti_tuple = tuple(
            [i for i in range(batch_size)]
        )  # Used for indexing a array down below, probably a better way to do this
        nstep_final_gamma = nstep_gamma**10

        # Samples stored as a list of dictionaries. have to get the samples from that dict list
        exp_minibatch = expert_buffer.sample(batch_size)
        exp_zip_batch = []
        for i in exp_minibatch:
            exp_zip_batch.append(i['sample'])

        exp_states_batch, exp_action_batch, exp_reward_batch, exp_next_states_batch, \
        exp_done_batch, exp_nstep_rew_batch, exp_nstep_next_batch = map(np.array, zip(*exp_zip_batch))
        is_expert_input = np.ones((batch_size, 1))

        # Expert action made into a 2d array for when tf.gather_nd is called during training
        input_exp_action = np.zeros((batch_size, 2))
        input_exp_action[:, 0] = np.arange(batch_size)
        input_exp_action[:, 1] = exp_action_batch

        exp_margin = np.ones((batch_size, action_len)) * exp_margin_constant
        exp_margin[
            np.arange(batch_size),
            exp_action_batch] = 0.  # Expert chosen actions don't have margin
        next_states_batch = exp_next_states_batch
        nstep_next_batch = exp_nstep_next_batch
        states_batch = exp_states_batch
        q_values_next, nstep_q_values_next = sess.run(
            [model.dq_output, model.nstep_output],
            feed_dict={
                model.input_img_dq: next_states_batch,
                model.input_img_nstep: nstep_next_batch,
                model.actions: exp_action_batch,
                model.input_expert_action: empty_action_batch,
                model.input_is_expert: empty_batch_by_one,
                model.input_expert_margin: empty_batch_by_action_len
            })
        action_max = np.argmax(q_values_next, axis=1)
        nstep_action_max = np.argmax(nstep_q_values_next, axis=1)
        dq_targets, nstep_targets = sess.run(
            [model.dq_output, model.nstep_output],
            feed_dict={
                model.input_img_dq: states_batch,
                model.input_img_nstep: states_batch,
                model.actions: exp_action_batch,
                model.input_expert_action: empty_action_batch,
                model.input_is_expert: empty_batch_by_one,
                model.input_expert_margin: empty_batch_by_action_len
            })
        reward_batch = exp_reward_batch
        done_batch = exp_done_batch
        dq_targets[ti_tuple,exp_action_batch] = reward_batch + \
                                                 (1 - done_batch) * gamma \
                                                 * q_values_next[np.arange(batch_size),action_max]
        nstep_rew_batch = exp_nstep_rew_batch
        done_batch = exp_done_batch
        nstep_targets[ti_tuple,exp_action_batch] = nstep_rew_batch + \
                                                    (1 - done_batch) * nstep_final_gamma \
                                                    * nstep_q_values_next[np.arange(batch_size),nstep_action_max]

        action_batch = exp_action_batch
        dq_targets = dq_targets[np.arange(batch_size), action_batch]
        nstep_targets = nstep_targets[np.arange(batch_size), action_batch]
        _, loss_summary, td_error_dq, td_error_nstep, slmc_value = sess.run(
            [
                model.updateModel, model.summaries, model.td_error_dq,
                model.td_error_nstep, model.slmc_output
            ],
            feed_dict={
                model.input_img_dq: states_batch,
                model.input_img_nstep: states_batch,
                model.actions: exp_action_batch,
                model.input_expert_action: input_exp_action,
                model.input_is_expert: is_expert_input,
                model.input_expert_margin: exp_margin,
                model.targetQ_dq: dq_targets,
                model.targetQ_nstep: nstep_targets
            })

        summary_writer.add_summary(loss_summary, current_step)

        #dq_loss = td_error_dq
        #nstep_loss = td_error_nstep
        #sample_losses = dq_loss_weighted + nstep_loss_weighted + np.abs(slmc_value)
        #expert_buffer.update_weights(exp_minibatch, loss_summary)

        if (current_step % 1000 == 0):
            saver.save(
                sess,
                expert_model_path + '/model-' + str(current_step) + '.cptk')

    # 3) Train DQFD Model, starting from Expert Model
    print('Training DQFD model')
    env = gym.make(env_name)

    ckpt = tf.train.get_checkpoint_state(expert_model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    replay_buffer = replay.PrioritizedReplayBuffer(75000,
                                                   alpha=0.4,
                                                   beta=0.6,
                                                   epsilon=0.001)

    max_timesteps = 100000
    min_buffer_size = 5000
    epsilon_start = 0.99
    epsilon_min = 0.01
    nsteps = 10
    batch_size = 32
    expert_margin = 0.8
    gamma = 0.99
    nstep_gamma = 0.99

    update_every = 100  # update target_model after this many training steps
    time_int = int(time.time())  # for saving models

    nstep_state_deque = deque()
    nstep_action_deque = deque()
    nstep_nexts_deque = deque()
    nstep_done_deque = deque()

    nstep_rew_list = []
    empty_by_one = np.zeros((1, 1))
    empty_exp_action_by_one = np.zeros((1, 2))
    empty_action_len_by_one = np.zeros((1, action_len))

    episode_start_ts = 0  # when this reaches n_steps, can start populating n_step_maxq_deque

    explore_ts = max_timesteps * 0.8

    loss = np.zeros((4, ))
    epsilon = epsilon_start
    curr_obs = env.reset()
    curr_obs = curr_obs['pov']

    # paper samples expert and self generated samples by weights, I used fixed proportion like Ape-X DQfD
    exp_batch_size = int(batch_size / 4)
    gen_batch_size = batch_size - exp_batch_size
    episode = 1
    total_rew = 0.
    for current_step in range(max_timesteps):
        print("current_step: " + str(current_step))

        episode_start_ts += 1

        # get action
        if random.random() <= epsilon:
            action_index = random.randint(0, 12)
        else:
            #temp_curr_obs = np.array(curr_obs)
            #temp_curr_obs = temp_curr_obs.reshape(1, temp_curr_obs.shape[0], temp_curr_obs.shape[1], temp_curr_obs.shape[2])

            #print("temp_curr_obs.shape: " + str(temp_curr_obs.shape))
            #print("temp_curr_obs: " + str(temp_curr_obs))

            empty_action_by_one = np.zeros((1))
            q = sess.run(model.dq_output,
                         feed_dict={
                             model.input_img_dq: [curr_obs],
                             model.input_img_nstep: [curr_obs],
                             model.actions: empty_action_by_one,
                             model.input_expert_action:
                             empty_exp_action_by_one,
                             model.input_is_expert: empty_by_one,
                             model.input_expert_margin: empty_action_len_by_one
                         })
            #print("q: " + str(q))
            #q, _, _ = train_model.predict([temp_curr_obs, temp_curr_obs, empty_by_one, empty_exp_action_by_one, empty_action_len_by_one])
            action_index = np.argmax(q)

        # reduce exploration rate epsilon
        if epsilon > epsilon_min:
            epsilon -= (epsilon_start - epsilon_min) / explore_ts

        action = env.action_space.noop()
        if (action_index == 0):
            action['camera'] = [0, -5]
            action['attack'] = 0
        elif (action_index == 1):
            action['camera'] = [0, -5]
            action['attack'] = 1
        elif (action_index == 2):
            action['camera'] = [0, 5]
            action['attack'] = 0
        elif (action_index == 3):
            action['camera'] = [0, 5]
            action['attack'] = 1
        elif (action_index == 4):
            action['camera'] = [-5, 0]
            action['attack'] = 0
        elif (action_index == 5):
            action['camera'] = [-5, 0]
            action['attack'] = 1
        elif (action_index == 6):
            action['camera'] = [5, 0]
            action['attack'] = 0
        elif (action_index == 7):
            action['camera'] = [5, 0]
            action['attack'] = 1
        elif (action_index == 8):
            action['forward'] = 1
            action['attack'] = 0
        elif (action_index == 9):
            action['forward'] = 1
            action['attack'] = 1
        elif (action_index == 10):
            action['jump'] = 1
            action['attack'] = 0
        elif (action_index == 11):
            action['jump'] = 1
            action['attack'] = 1
        elif (action_index == 12):
            action['attack'] = 1

        # do action
        obs, rew, done, info = env.step(action)
        obs = obs['pov']
        #print("_rew: " + str(_rew))

        # reward clip value from paper = sign(r) * log(1+|r|)
        rew = np.sign(rew) * np.log(1. + np.abs(rew))
        #print("_rew: " + str(_rew))
        total_rew += rew
        #print(action_command, _rew, epsilon)
        nstep_state_deque.append(curr_obs)
        nstep_action_deque.append(action_index)
        nstep_nexts_deque.append(obs)
        nstep_done_deque.append(done)
        nstep_rew_list.append(rew)
        if episode_start_ts > 10:
            add_transition(replay_buffer, nstep_state_deque,
                           nstep_action_deque, nstep_rew_list,
                           nstep_nexts_deque, nstep_done_deque, obs, False,
                           nsteps, nstep_gamma)

        if (current_step % 1000 == 0):
            print("total_rew: " + str(total_rew))
            print("epsilon: " + str(epsilon))
            print("")

            saver.save(
                sess,
                dqfd_model_path + '/model-' + str(current_step) + '.cptk')
            #nstep_rew_mean = sum(nstep_rew_list) / len(nstep_rew_list)
            #print("nstep_rew_mean: " + str(nstep_rew_mean))

        # if episode done we reset
        if done:
            #summary_writer.add_summary(nstep_rew_mean, current_step)
            #print('episode done {}'.format(total_rew))
            # emptying the deques
            add_transition(replay_buffer, nstep_state_deque,
                           nstep_action_deque, nstep_rew_list,
                           nstep_nexts_deque, nstep_done_deque, obs, True,
                           nsteps, nstep_gamma)

            # reset the environment, get the current state
            curr_obs = env.reset()
            curr_obs = curr_obs['pov']

            nstep_state_deque.clear()
            nstep_action_deque.clear()
            nstep_rew_list.clear()
            nstep_nexts_deque.clear()
            nstep_done_deque.clear()

            episode_start_ts = 0
        else:
            curr_obs = obs  # resulting state becomes the current state

        # train the network using expert and experience replay
        # I fix the sample between the two while paper samples based on priority
        if current_step > min_buffer_size:
            # sample from expert and experience replay and concatenate into minibatches
            # get target network and train network predictions
            # use Double DQN
            exp_minibatch = expert_buffer.sample(exp_batch_size)
            exp_zip_batch = []
            for i in exp_minibatch:
                exp_zip_batch.append(i['sample'])

            exp_states_batch, exp_action_batch, exp_reward_batch, exp_next_states_batch, \
            exp_done_batch, exp_nstep_rew_batch, exp_nstep_next_batch = map(np.array, zip(*exp_zip_batch))

            is_expert_input = np.zeros((batch_size, 1))
            is_expert_input[0:exp_batch_size, 0] = 1

            # expert action made into a 2d array for when tf.gather_nd is called during training
            input_exp_action = np.zeros((batch_size, 2))
            input_exp_action[:, 0] = np.arange(batch_size)
            input_exp_action[0:exp_batch_size, 1] = exp_action_batch
            expert_margin = np.ones((batch_size, action_len)) * expert_margin
            expert_margin[
                np.arange(exp_batch_size),
                exp_action_batch] = 0.  #expert chosen actions don't have margin

            minibatch = replay_buffer.sample(gen_batch_size)
            zip_batch = []
            for i in minibatch:
                zip_batch.append(i['sample'])

            states_batch, action_batch, reward_batch, next_states_batch, done_batch, \
                nstep_rew_batch, nstep_next_batch = map(np.array, zip(*zip_batch))

            #print("exp_states_batch.shape: " + str(exp_states_batch.shape))
            #print("states_batch.shape: " + str(states_batch.shape))

            # concatenating expert and generated replays
            concat_states = np.concatenate((exp_states_batch, states_batch),
                                           axis=0)
            concat_next_states = np.concatenate(
                (exp_next_states_batch, next_states_batch), axis=0)
            concat_nstep_states = np.concatenate(
                (exp_nstep_next_batch, nstep_next_batch), axis=0)
            concat_reward = np.concatenate((exp_reward_batch, reward_batch),
                                           axis=0)
            concat_done = np.concatenate((exp_done_batch, done_batch), axis=0)
            concat_action = np.concatenate((exp_action_batch, action_batch),
                                           axis=0)
            concat_nstep_rew = np.concatenate(
                (exp_nstep_rew_batch, nstep_rew_batch), axis=0)

            empty_batch_by_one = np.zeros((batch_size, 1))
            empty_action_batch = np.zeros((batch_size, 2))
            empty_action_batch[:, 0] = np.arange(batch_size)
            empty_batch_by_action_len = np.zeros((batch_size, action_len))
            ti_tuple = tuple(
                [i for i in range(batch_size)]
            )  # Used for indexing a array down below, probably a better way to do this
            nstep_final_gamma = nstep_gamma**10

            next_states_batch = concat_next_states
            nstep_next_batch = concat_nstep_states
            states_batch = concat_states
            action_batch = concat_action
            reward_batch = concat_reward
            nstep_rew_batch = concat_nstep_rew
            done_batch = concat_done

            q_values_next, nstep_q_values_next = sess.run(
                [model.dq_output, model.nstep_output],
                feed_dict={
                    model.input_img_dq: next_states_batch,
                    model.input_img_nstep: nstep_next_batch,
                    model.actions: action_batch,
                    model.input_expert_action: empty_action_batch,
                    model.input_is_expert: empty_batch_by_one,
                    model.input_expert_margin: empty_batch_by_action_len
                })

            action_max = np.argmax(q_values_next, axis=1)
            nstep_action_max = np.argmax(nstep_q_values_next, axis=1)
            dq_targets, nstep_targets = sess.run(
                [model.dq_output, model.nstep_output],
                feed_dict={
                    model.input_img_dq: states_batch,
                    model.input_img_nstep: states_batch,
                    model.actions: action_batch,
                    model.input_expert_action: empty_action_batch,
                    model.input_is_expert: empty_batch_by_one,
                    model.input_expert_margin: empty_batch_by_action_len
                })

            dq_targets[ti_tuple,action_batch] = reward_batch + \
                                                     (1 - done_batch) * gamma \
                                                     * q_values_next[np.arange(batch_size),action_max]
            nstep_targets[ti_tuple,action_batch] = nstep_rew_batch + \
                                                        (1 - done_batch) * nstep_final_gamma \
                                                        * nstep_q_values_next[np.arange(batch_size),nstep_action_max]

            dq_targets = dq_targets[np.arange(batch_size), action_batch]
            nstep_targets = nstep_targets[np.arange(batch_size), action_batch]
            _, loss_summary, slmc_value = sess.run(
                [model.updateModel, model.summaries, model.slmc_output],
                feed_dict={
                    model.input_img_dq: states_batch,
                    model.input_img_nstep: states_batch,
                    model.actions: action_batch,
                    model.input_expert_action: input_exp_action,
                    model.input_is_expert: is_expert_input,
                    model.input_expert_margin: expert_margin,
                    model.targetQ_dq: dq_targets,
                    model.targetQ_nstep: nstep_targets
                })

            summary_writer.add_summary(loss_summary, current_step)
            #print("slmc_value: " + str(slmc_value))
            #dq_loss = td_error_dq
            #nstep_loss = td_error_nstep
            #sample_losses = np.abs(td_error_dq)

            #expert_buffer.update_weights(exp_minibatch, sample_losses[:exp_batch_size])
            #replay_buffer.update_weights(minibatch, sample_losses[-(batch_size - exp_batch_size):])

    print('Test DQFD model')
    ckpt = tf.train.get_checkpoint_state(expert_model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    epsilon = 0.01
    obs = env.reset()
    s = obs['pov']
    total_rew = 0
    while True:
        if random.random() <= epsilon:
            action_index = random.randint(0, action_len - 1)
        else:
            q = sess.run(model.dq_output, feed_dict={model.input_img_dq:
                                                     [s]})[0]
            #print("q: " + str(q))

            action_index = np.argmax(q)
            #action_index = 0
            #print("action_index: " + str(action_index))
            #print("")

        action = env.action_space.noop()
        if (action_index == 0):
            action['camera'] = [0, -5]
            action['attack'] = 0
        elif (action_index == 1):
            action['camera'] = [0, -5]
            action['attack'] = 1
        elif (action_index == 2):
            action['camera'] = [0, 5]
            action['attack'] = 0
        elif (action_index == 3):
            action['camera'] = [0, 5]
            action['attack'] = 1
        elif (action_index == 4):
            action['camera'] = [-5, 0]
            action['attack'] = 0
        elif (action_index == 5):
            action['camera'] = [-5, 0]
            action['attack'] = 1
        elif (action_index == 6):
            action['camera'] = [5, 0]
            action['attack'] = 0
        elif (action_index == 7):
            action['camera'] = [5, 0]
            action['attack'] = 1
        elif (action_index == 8):
            action['forward'] = 1
            action['attack'] = 0
        elif (action_index == 9):
            action['forward'] = 1
            action['attack'] = 1
        elif (action_index == 10):
            action['jump'] = 1
            action['attack'] = 0
        elif (action_index == 11):
            action['jump'] = 1
            action['attack'] = 1
        elif (action_index == 12):
            action['attack'] = 1

        #print("action: " + str(action))
        #print("")

        obs, rew, done, info = env.step(action)
        s1 = obs['pov']
        total_rew += rew
        #print("total_rew: " + str(total_rew))
        s = s1

        #env.render()
        if done:
            print("total_rew: " + str(total_rew))
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
