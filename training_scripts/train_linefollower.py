#!/usr/bin/env python3
import random
import gym_gazebo
import gymnasium as gym
import numpy as np
import pickle
import sys
from torch.utils.tensorboard.writer import SummaryWriter

def main():
    writer = SummaryWriter()
    env = gym.make('LineFollowerEnv-v0')
    observation_space = env.observation_space
    action_space = env.action_space

    epsilon = 0.8
    epsilon_decay = 0.99

    if len(sys.argv) > 1 and sys.argv[1] == "load":
        print(f"Loading Q-table from {sys.argv[2]}")
        q_table = loadQ(sys.argv[2])
    else:
        q_table = np.zeros((observation_space.n, action_space.n))

    highest_reward = 0

    try:
        print(3)
        state, _ = env.reset()
        print(4)
        for episode in range(1000):
            total_reward = 0
            actions_per_episode = 0
            done = False
            print(5)
            while not done:
                actions_per_episode += 1
                action = choose_action(state, q_table, epsilon, action_space)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                done = terminated or truncated

                q_table = learn(state, action, next_state, reward, q_table)

                state = next_state
                total_reward += reward

            # Log data
            print(f"Episode {episode}: Epsilon: {epsilon} | Total Reward: {total_reward}")
            print("Q values:")
            print(q_table)
            writer.add_scalar("Total Reward", total_reward, episode)
            writer.add_scalar("Actions per episode", actions_per_episode, episode)

            if total_reward > highest_reward:
                highest_reward = total_reward
                print("New highest reward: {total_reward}.")
                saveQ(q_table, "best_q_table.pkl")

            env.reset()
            epsilon = max(0.01, epsilon * epsilon_decay)  # Decay epsilon

    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    finally:
        env.close()
        writer.flush()
        writer.close()


def learn(state, action, next_state, reward, q_table):
    alpha = 0.1
    gamma = 0.9

    q_table[state, action] += alpha*(reward+ gamma*np.max(q_table[next_state, :]) - q_table[state, action])

    return q_table


def choose_action(state, q_table, epsilon, action_space):
    action = 0

    if random.random() > epsilon:  # greedy
        return round(np.argmax(q_table[state, :]))

    return action_space.sample()  # random


def saveQ(q_table, save_path):
    print(f"Saving Q-table to {save_path}")

    with open(save_path, 'wb') as file:
        pickle.dump(q_table, file)

    return

def loadQ(load_path):

    with open(load_path, 'rb') as file:
        q_table = pickle.load(file)

    return q_table


if __name__ == "__main__":
    main()