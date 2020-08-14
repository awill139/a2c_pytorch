import gym
import argparse
import matplotlib.pyplot as plt

from train import train
from a2c import ActorCritic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required = False, default = 'LunarLander-v2')
    parser.add_argument('--hidden', required = False, default = 256)
    parser.add_argument('--max_episodes', required = False, default = 3000)
    parser.add_argument('--num_steps', required = False, default = 300)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    num_steps = args.num_steps
    max_episodes = args.max_episodes
    hidden = args.hidden

    env = gym.make(args.env)
    
    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.n
    
    ac = ActorCritic(num_inputs = n_inputs, num_acts = n_outputs, hidden = hidden)

    rewards, losses, avg_len, all_len = train(model = ac, env = env, episodes = max_episodes, num_steps = num_steps)

    
if __name__ == '__main__':
    main()
