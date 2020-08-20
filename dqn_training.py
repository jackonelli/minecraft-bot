from pathlib import Path
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
from dqfd_navigate import Qnetwork, add_transition, parse_demo


def main():
    env_name = "MineRLNavigateDense-v0"
    root_path = Path(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(
        root_path, 'data')  # Or wherever you have stored the expert data

    summary_path = root_path / 'train_summary' / env_name

    dqfd_model_path = root_path / 'dqfd_model'
    expert_model_path = dqfd_model_path
    mission_name = "dense_double_stair.xml"
    env = gym.make(env_name,
                   xml=str(Path(root_path) / "my_mission_xmls" / mission_name))
    max_episode_steps = 500
    action_len = 13
    train_steps = 1e5
    batch_size = 32
    gamma = 0.99
    nstep_gamma = 0.99
    exp_margin_constant = 0.8

    expert_buffer = replay.PrioritizedReplayBuffer(75000,
                                                   alpha=0.4,
                                                   beta=0.6,
                                                   epsilon=0.001)
    expert_buffer = parse_demo(env_name, expert_buffer, data_path)

    model = Qnetwork()

    saver = tf.compat.v1.train.Saver()
    trainables = tf.compat.v1.trainable_variables()
    init = tf.compat.v1.global_variables_initializer()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(init)

    summary_writer = tf.compat.v1.summary.FileWriter(summary_path)
    time_int = int(time.time())
    loss = np.zeros((4, ))

    env = gym.make(env_name)

    ckpt = tf.train.get_checkpoint_state(expert_model_path)
    saver.restore(sess, str(Path(expert_model_path) / "model-11000.cptk"))

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
    steps_in_current_epi = 0
    total_rew = 0.
    for current_step in range(max_timesteps):

        episode_start_ts += 1

        steps_in_current_epi += 1
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

        if (current_step % update_every == 0):
            print("total_rew: " + str(total_rew))
            print("epsilon: " + str(epsilon))
            print("")

            saver.save(
                sess,
                str(dqfd_model_path /
                    ('model-' + str(current_step) + '.cptk')))
            #nstep_rew_mean = sum(nstep_rew_list) / len(nstep_rew_list)
            #print("nstep_rew_mean: " + str(nstep_rew_mean))

        # if episode done we reset
        if done or steps_in_current_epi > max_episode_steps:
            print("Resetting env")
            steps_in_current_epi = 0
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
