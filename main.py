"""Main script"""
import logging
import minerl
import gym


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO)
    #experiment_name = "MineRLObtainDiamondVectorObf-v0"
    experiment_name = "MineRLNavigateDense-v0"
    data_path = "data"
    minerl.data.download(directory=data_path, experiment=experiment_name)
    env = gym.make(experiment_name)

    obs = env.reset()
    done = False

    step = 0
    net_reward = 0
    while not done:
        step += 1
        logging.info("Step: {}".format(step))
        action = env.action_space.noop()

        action['camera'] = [0, 0.03 * obs["compassAngle"]]
        action['back'] = 0
        action['forward'] = 1
        action['jump'] = 1
        action['attack'] = 1

        obs, reward, done, info = env.step(action)

        net_reward += reward
        print("Total reward: ", net_reward)


if __name__ == "__main__":
    main()
