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
from dqfd_navigate import Qnetwork


def action_from_q_val(action_index, action):
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
    return action


def main():
    env_name = "MineRLNavigate-v0"
    root_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        root_path, 'data')  # Or wherever you have stored the expert data

    summary_path = root_path + 'train_summary/' + env_name

    dqfd_model_path = root_path + 'dqfd_model'
    expert_model_path = root_path + '/expert_model'

    action_len = 13
    train_steps = 1e5
    batch_size = 32
    gamma = 0.99
    nstep_gamma = 0.99
    exp_margin_constant = 0.8

    # 1) Get Expert Data
    # expert_buffer = replay.PrioritizedReplayBuffer(75000,
    #                                                alpha=0.4,
    #                                                beta=0.6,
    #                                                epsilon=0.001)
    # expert_buffer = parse_demo(env_name, expert_buffer, data_path)

    # # 2) Train Expert Model on Data
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

    env = gym.make(env_name,
                   xml=str(
                       Path(root_path) / "my_mission_xmls" /
                       "outsideToEntranceHard.xml"))

    saver.restore(sess, str(Path(expert_model_path) / "model-50000.cptk"))
    obs = env.reset()
    done = False

    step = 0
    net_reward = 0
    obs = env.reset()
    while not done:
        step += 1
        q_value = sess.run(model.dq_output,
                           feed_dict={model.input_img_dq: [obs["pov"]]})[0]
        action = action_from_q_val(np.argmax(q_value), env.action_space.noop())

        # action['camera'] = [0, 0.03 * obs["compassAngle"]]
        # action['back'] = 0
        # action['forward'] = 1
        # action['jump'] = 1
        # action['attack'] = 1

        obs, reward, done, info = env.step(action)

        net_reward += reward
        print("Step: {}, action: {}, total reward: {} ".format(
            step, np.argmax(q_value), net_reward))

    env.close()


if __name__ == "__main__":
    main()
