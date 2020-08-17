"""Main script"""
from pathlib import Path
import logging
import minerl
import gym


def main():
    """Main entry point"""
    logging.basicConfig(level=logging.DEBUG)
    experiment_name = "MineRLObtainDiamondVectorObf-v0"
    data_path = Path("data")
    minerl.data.download(directory=data_path, experiment=experiment_name)
    env = gym.make(experiment_name)
    # obs = env.reset()
    # print(obs)


if __name__ == "__main__":
    main()
