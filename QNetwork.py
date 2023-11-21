"""Deep Q Network (DQN) Agent for the game of Othello."""

import torch
import torch.nn as nn
import random

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): 64  # 8x8 board
            action_size (int): 64  # highest possible number of actions 
                        (shouldn't ever be this high, should be 64 - 4)
            seed (int): random seed  # 8x8 board
            fc1_units (int): number of nodes in the first hidden layer
            fc2_units (int): number of nodes in the second hidden layer"""
        super(QNetwork, self).__init__()

        if seed == 0:
            self.seed = torch.manual_seed(random.randint(0, 1000))
        else:
            self.seed = torch.manual_seed(seed)
        print("seed in QNetwork", self.seed)

        self.fc1 = nn.Linear(state_size, fc1_units) # Input layer
        self.fc2 = nn.Linear(fc1_units, fc2_units) # Hidden layer
        self.fc3 = nn.Linear(fc2_units, action_size)  # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        """Build a network that maps state -> action values.
        Defines a forward pass of the NN
        Takes in a state and returns the action values for that state based on the ReLu activation function
        Output of this function is the action values for the state and the output of the NN

        Params
        ======
           x (array_like): state


        Returns
        =======
            x (array_like): action values for the state"""

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    print("QNetwork.py")