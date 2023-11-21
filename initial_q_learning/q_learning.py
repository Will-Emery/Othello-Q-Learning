#import othello as othello
import gym
import random
import numpy as np
import pandas as pd
from othello import OthelloEnv

class QLearning:
    def init(self, learning_mode = False, q_table_file = "best_table.csv"):
        """Initializes the Q-learning algorithm.
        
        Parameters:
            learning_mode: Whether or not the algorithm should learn
            q_table_file: The file to load the Q-table from if learning_mode is False
        Returns:
            None"""
        

        self.learning_mode = learning_mode
        self.passed_q_table = q_table_file
        self.define_states()


        if self.learning_mode == False:
            # Load the Q-table from a file
            self.q_table = np.loadtxt(q_table_file, delimiter=",")
        if self.learning_mode == True:
            self.learning_df = pd.DataFrame(columns=["episode", "epsilon", "total_reward"])
        if self.learning_mode == None:
            # Load the Q-table from a file
            # this mode is for debugging
            self.q_table = np.loadtxt(q_table_file, delimiter=",")

        # Set the learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1


    def define_states(self):
        '''Defines the states of the game. This will be done based on the board state.'''
        # Define the states there are 64 possible positions, with each position holding a value of -1, 0, or 1
        # This means that there are 3^64 possible states with a possible of max 64 moves per state
        # This means that there are 3^64 * 50 possible states
        # A state/action is defined as row_col_locationState_action, where action is also defined as actRow_actCol


        #NOTE: State Breakdown
        # row and col = 0 - 7
        # locationState = 0 - 2 where 0 is empty and 1 is player -1 and 2 is player 1
        # actionRow and actionCol = 0 - 7

        # Define the states
        self.states = []
        for row in range(8):
            for col in range(8):
                for locationState in range(3):
                    for actionRow in range(8):
                        for actionCol in range(8):
                            self.states.append(str(row) + "_" + str(col) + "_" + str(locationState))

        # Now to define the actions, the player can only place stone in a location
        self.actions = []
        for row in range(8):
            for col in range(8):
                self.actions.append(str(row) + "_" + str(col))


        #Q-Table
        #rows are the states, defined as row_col_locationState
        #columns are the actions, defined as actRow_actCol
        self.qtable = np.zeros((len(self.states), len(self.actions)))

        if self.learning_mode == False:
            # Load the Q-table from a file
            self.q_table = np.loadtxt(self.passed_q_table, delimiter=",")

    def get_state_index(self, state):
        '''Gets the index of the state q_table
        
        Parameters:
            state: The state to get the index of
        Columns:
            The index of the state'''
        return self.states.index(state)
                
        


        





if __name__ == "__main__":
    env = OthelloEnv()

    # Reset the environment
    board = env.reset()

    # Run the game loop
    while not env.done:
        # Get the legal moves
        legal_moves = env.legal_moves()

        # Choose a random move
        move = random.choice(legal_moves)

        # Make the move
        board, reward, done, info = env.step(move)

        # Print the board
        print(board)
        print()

        # Check if the game is over
        if env.is_game_over():
            env.done = True
