"""Module: DeepQ_learning

This module contains the Deep Q-learning algorithm for the Othello game."""

import gym
from gym import spaces
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from QNetwork import QNetwork as qn
from othello import OthelloEnv
import matplotlib.pyplot as plt
import pandas as pd
import time

class QLearningAgent:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon=0.5):
        self.q_network = qn(input_size, output_size)
        self.output_size = output_size
        self.target_q_network = qn(input_size, output_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.01)
        self.gamma = gamma
        self.epsilon = epsilon

        self.random_actions_taken = 0

    def epsilon_greedy_action(self, state, legal_moves):
        """
        Choose an action using an epsilon-greedy strategy.

        Parameters:
            state (np.ndarray): The current state of the game.
            legal_moves (list): List of legal moves in the format [[row, col], [row, col], ...].

        Returns:
            action (list): Chosen action in the format [row, col].
        """
        if np.random.rand() < self.epsilon:
            print("random action chosen")
            self.random_actions_taken += 1
            return random.choice(legal_moves)
        else:
            flattened_state = state.flatten()
            q_values = self.q_network(torch.tensor(flattened_state, dtype=torch.float32))
            legal_q_values = q_values[[self.coordinate_to_index(move) for move in legal_moves]]
            max_legal_index = torch.argmax(legal_q_values).item()
            return legal_moves[max_legal_index]

    def train(self, state, action, reward, next_state, legal_moves, done):
        """
        Train the Q-learning agent.

        Parameters:
            state (np.ndarray): The current state of the game.
            action (list): Chosen action in the format [row, col].
            reward (float): The reward received after taking the action.
            next_state (np.ndarray): The next state of the game.
            legal_moves (list): List of legal moves in the format [[row, col], [row, col], ...].
            done (bool): True if the episode is finished, False otherwise.
        """
        state = state.flatten()
        next_state = next_state.flatten()

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.q_network(state)
        next_q_values = self.target_q_network(next_state)

        target_q_value = reward + (1 - done) * self.gamma * torch.max(next_q_values)

        loss = F.mse_loss(q_values[self.coordinate_to_index(action)], target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update the target Q-network with the current Q-network's parameters
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def coordinate_to_index(self, coordinate):
        """
        Convert a coordinate [row, col] to an index in the Q-values array.

        Parameters:
            coordinate (list): Coordinate in the format [row, col].

        Returns:
            index (int): Index in the Q-values array.
        """
        row, col = coordinate
        return row * 8 + col
    
def plot_data(df):
    """
    Plot the data gathered from the training loop.

    Parameters:
        df (pd.DataFrame): Dataframe containing the data to plot.
    """
    # Convert columns to numeric type
    df['episode'] = pd.to_numeric(df['episode'])
    df['total_reward'] = pd.to_numeric(df['total_reward'])

    plt.plot(df['episode'], df['total_reward'], label='Total Reward')
    
    # Perform linear regression
    coefficients = np.polyfit(df['episode'], df['total_reward'], 1)
    trendline = np.polyval(coefficients, df['episode'])
    
    # Plot the trendline
    plt.plot(df['episode'], trendline, label='Trendline', color='red')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs. Episode')
    plt.legend()
    plt.show()

    plt.plot(df['episode'], df['random_actions_taken'])
    plt.xlabel('Episode')
    plt.ylabel('Random Actions Taken')
    plt.title('Random Actions Taken vs. Episode')
    plt.show()

def graph_win_loss(win_percentage):
    """
    Make a pie chart of the win/loss ratio.
    """
    labels = 'Wins', 'Losses'
    sizes = [win_percentage, 1 - win_percentage]
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


if __name__ == "__main__":
    #start a timer
    start_time = time.time()

    graphing_df = pd.DataFrame(columns=['episode', 'total_reward', 'random_actions_taken'])

    othello_env = OthelloEnv(1)

    # Initialize the Q-learning agent
    input_size = 64  # Define the size of the state space
    output_size = 64  # Define the number of possible actions
    q_agent = QLearningAgent(input_size, output_size, gamma=0.9, epsilon=0.9)

    num_episodes = 1000  # Define the number of episodes to train for
    target_update_frequency = 1  # Define the frequency for updating the target network

    decay_rate = q_agent.epsilon / num_episodes

    # Training loop
    for episode in range(num_episodes):
        state = othello_env.reset()
        print("resetting for next episode")
        done = False
        total_reward = 0  # Initialize total reward for the episode
        q_agent.random_actions_taken = 0 # initialize random actions taken for the episode


        while not done:
            # Get the legal moves for the current state
            legal_moves = othello_env.legal_moves()

            # Choose an action using epsilon-greedy strategy
            action = q_agent.epsilon_greedy_action(state, legal_moves)

            # Take the action and observe the next state and reward
            next_state, reward, legal_moves, done = othello_env.step(action)

            # Train the Q-learning agent
            q_agent.train(state, action, reward, next_state,legal_moves, done)

            # Update the current state
            state = next_state

            # Accumulate the total reward for the episode
            total_reward += reward

            if done:
                print("Episode: {}, Total Reward: {}".format(episode, total_reward))
                # add the episode, total reward, and random actions taken to the graphing dataframe
                graphing_df = graphing_df._append({'episode': episode, 'total_reward': total_reward, 'random_actions_taken': q_agent.random_actions_taken}, ignore_index=True)

        # Optionally update the target Q-network periodically
        if episode % target_update_frequency == 0:
            q_agent.update_target_network()
        
        # Decay epsilon
        q_agent.epsilon -= decay_rate

    #plot the data gathered
    plot_data(graphing_df)

    #test the trained agent
    num_test_games = 1000
    wins = 0
    for _ in range(num_test_games):
        print("test game: ", _)
        state = othello_env.reset()
        done = False

        while not done:
            legal_moves = othello_env.legal_moves()
            action = q_agent.epsilon_greedy_action(state, legal_moves)
            next_state, _, _, done = othello_env.step(action)
            state = next_state

        if othello_env.winner == 1:  # Agent's player index is 1
            wins += 1

    print("Winning rate after {} test games: {:.2%}".format(num_test_games, wins / num_test_games))
    print("Time taken: ", time.time() - start_time)
    graph_win_loss(wins / num_test_games)