"""Module: othello.py
Author: William Emery
Date: 11/10/23

This module contains my attampt at an implementation of the Othello environment."""

import gym
from gym import spaces
import random
import numpy as np

class OthelloEnv(gym.Env):
    def __init__(self, num_players=2):
        """Initializes the Othello environment.
        
        Params:
        ======
            board_dimensions: The dimensions of the board as a tuple (rows, columns)
        Returns:
        ======
            None"""
        super(OthelloEnv, self).__init__()

        # Define the action and observation space
        self.action_space = spaces.Discrete(64) # 8x8 board
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8, 8), dtype=np.int8)

        # Initialize the board
        self.board = np.zeros((8, 8), dtype=np.int8)

        # Set the initial game state
        self.current_player = 1
        self.done = False
        self.turn_number = 0
        self.num_players = num_players

        self.winner = None
        
        print("num_players: " + str(self.num_players))


    def reset(self):
        """Resets the board to its initial state.
        
        Params:
        ======
            None
        Returns:
        ======
            The board in its initial state."""
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.board[3, 3] = self.board[4, 4] = 1
        self.board[3, 4] = self.board[4, 3] = -1
        self.current_player = 1
        self.done = False
        self.turn_number = 0
        self.winner = None


        return self.board
    
    
    def render(self):
        """Prints the board.
        
        Params:
        ======
            None
            
        Returns:
        ======
            None"""
        print(self.board)
    

    def is_game_over(self):
        """Checks if the game is over.
        
        Params:
        ======
            None
        
        Returns:
        ======
            True if the game is over, False otherwise."""
        # The game is over if the board is full or neither player can make a legal move.
        if(np.all(self.board != 0) or not self.legal_moves()):
            self.winner = self.get_winner()
            # print("Winner: " + str(winner))
            return True
        else:
            return False
        
    
    def get_winner(self):
        """Gets the winner of the game.

        Returns:
        ======
            1 if player 1 won, -1 if player 2 won, 0 if it was a tie."""
        player_1_score = np.sum(self.board == 1)
        player_2_score = np.sum(self.board == -1)

        if player_1_score > player_2_score:
            return 1
        elif player_2_score > player_1_score:
            return -1
        else:
            return 0
        

    def legal_moves(self):
        """Finds all the legal moves on the board.
        
        Params:
        ======
            None
            
        Returns:
        ======
            A list of all the legal moves on the board."""
        #legal moves are all the empty spaces that are next to an an occupied space
        legal_moves_list = []
        for row in range(8):
            for col in range(8):
                if self.board[row, col] == 0 and self.is_next_to_occupied(row, col):
                    legal_moves_list.append((row, col))
    
        return legal_moves_list
    

    def is_next_to_occupied(self, row, col):
        """Checks if a space is next to an occupied space.
        
        Params:
        ======
            row: The row of the space to check.
            col: The column of the space to check.
            
        Returns:
        ======
            True if the space is next to an occupied space, False otherwise."""
        #check all directions, if one of these is 1 or -1 then the move is valid
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                if row + dr < 0 or row + dr >= 8 or col + dc < 0 or col + dc >= 8:
                    continue
                if self.board[row + dr, col + dc] == 1 or self.board[row + dr, col + dc] == -1:
                    return True
                

    def step(self, action):
        """Function that steps forward the game by one turn
        
        Params:
        ======
            action: The action to take
            
        Returns:
        ======
            The new board, 
            the reward,
            game over,
            and the info"""
        # legal_moves_list = self.legal_moves()
        # # print("current player: " + str(self.current_player)
        # #         + " turn number: " + str(self.turn_number))
        # # print("legal moves: " + str(legal_moves_list))
        # # print("action: " + str(action))
        
        # if action not in legal_moves_list:
        #     return self.board, -1, True, {}
        
        # self.board[action[0], action[1]] = self.current_player
        # self.flip_pieces(action[0], action[1])
        
        # # Update Vars to reflect the new state
        # self.turn_number += 1
        # self.current_player *= -1

        # self.done = self.is_game_over()

        # return self.board, 0, self.done, {}

        if self.num_players == 2:
            return self.step_2_players(action)
        elif self.num_players == 1:
            return self.step_1_player(action)
        elif self.num_players == 0:
            return self.step_no_players(action)
        else:
            raise ValueError("Unsupported number of players: {}".format(self.num_players))

    def step_2_players(self, action):
        """Function that steps forward the game by one turn for 2 players
        
        Params:
        ======
            action: The action to take
            
        Returns:
        ======
            The new board, 
            the reward,
            game over"""
        legal_moves_list = self.legal_moves()

        if action not in legal_moves_list:
            # print("Invalid action")
            #get the action from the user
            action = random.choice(self.legal_moves())
            print("random action chosen as: " + str(action))
            # print("punishing the current player")
            return self.board, -5, self.done

        self.board[action[0], action[1]] = self.current_player
        reward = self.flip_pieces(action[0], action[1])

        self.turn_number += 1
        self.current_player *= -1

        self.done = self.is_game_over()

        return self.board, reward, self.legal_moves(), self.done

    def step_1_player(self, action):
        """Function that steps forward the game by one turn for 1 player
        
        Params:
        ======
            action: The action to take
            
        Returns:
        ======
            The new board,
            the reward,
            game over,
            and the info"""
        # Only allow the current player to take a step
        # print("current player action: " + str(action))
        return_result = self.step_2_players(action)
        return_reward = return_result[1]
        
        #check that current player is == -1 if not flip it
        if self.current_player == 1:
            # print("current player flipped to -1")
            self.current_player *= -1
        # If the game is over then return the result
        if self.done:
            return return_result
        
        # If the game is not over then let the other player take a step this is the random player
        legal_moves_list = self.legal_moves()
        action = random.choice(legal_moves_list)
        # print("random player action: " + str(action))
        self.board[action[0], action[1]] = self.current_player
        self.flip_pieces(action[0], action[1])
        self.turn_number += 1
        self.current_player *= -1
        self.done = self.is_game_over()

        return self.board, return_reward, self.legal_moves, self.done
    

    def step_no_players(self, action):
        # No players, simply choose random actions for both players

        # Player 1
        legal_moves_list = self.legal_moves()
        action = random.choice(legal_moves_list)
        self.board[action[0], action[1]] = self.current_player
        self.flip_pieces(action[0], action[1])
        self.turn_number += 1
        self.current_player *= -1

        # Player 2
        legal_moves_list = self.legal_moves()
        action = random.choice(legal_moves_list)
        self.board[action[0], action[1]] = self.current_player
        self.flip_pieces(action[0], action[1])
        self.turn_number += 1
        self.current_player *= -1

        self.done = self.is_game_over()
        return self.board
    
    
    def log_game_to_file(self, filename = "game_log.txt"):
        """Function that logs the game to a file
        
        Params:
        ======
            filename: The name of the file to log to
            
        Returns:
        ======
            None"""
        with open(filename, "a") as f:
            f.write("Turn Number " + str(self.turn_number) + "\n")
            f.write("Current Player: " + str(self.current_player) + "\n")
            f.write(str(self.board) + "\n")
            f.write("\n")


    def flip_pieces(self, start_row, start_col):
        """Function that flips the pieces of the board
        Large sections of this function are commented out because of their use in debugging.
        I am still not 100% confident that this function works as intended.

        Params:
        ======
            start_row: The row of the piece to flip
            start_col: The column of the piece to flip

        Returns:
        ======
            Reward: as an int for flipping, +1 for every piece flipped, if no pieces are flipped then -1
        """
        reward = -1  # Default reward if no pieces are flipped
        # highlighted_spaces = []  # List to store information about spaces for color highlighting

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                # If the direction is 0, 0 then skip it
                if dr == 0 and dc == 0:
                    continue

                row, col = start_row + dr, start_col + dc

                # If the direction is out of bounds then skip it
                if not (0 <= row < 8 and 0 <= col < 8):
                    continue

                # If the space is the opposite player then examine following spaces
                if self.board[row, col] == self.current_player * -1:
                    spaces_to_flip = []

                    # Continue checking until you reach the current player or an empty space
                    while 0 <= row < 8 and 0 <= col < 8 and (self.board[row, col] == (self.current_player * -1) or self.board[row, col] == 0):
                        # Add the space to flip to the list
                        spaces_to_flip.append((row, col))

                        # Move to the next space in the path
                        row += dr
                        col += dc

                    # If the last space in the path is the current player then flip all the pieces in the path
                    if 0 <= row < 8 and 0 <= col < 8 and self.board[row, col] == self.current_player:
                        reward = len(spaces_to_flip)  # Set reward to the number of pieces flipped
                        # highlighted_spaces.extend(spaces_to_flip)  # Add flipped spaces to the list
                        
                        #now iterate over the spaces_to_flip list and set those spaces on the board to the current player
                        for space in spaces_to_flip:
                            self.board[space[0], space[1]] = self.current_player

        # # Print statements with color highlighting
        # for r in range(8):
        #     for c in range(8):
        #         current_space = (r, c)
                
        #         # Highlight start_row and start_col in red
        #         if current_space == (start_row, start_col):
        #             print("\033[91m{}\033[0m".format(self.board[r, c]), end=" ")
        #         else:
        #             # Highlight everything they are checking in blue
        #             if current_space in highlighted_spaces:
        #                 print("\033[94m{}\033[0m".format(self.board[r, c]), end=" ")
        #             else:
        #                 print(self.board[r, c], end=" ")

        #         # Move to the next line after each row
        #     print()
        # print()  # Add an extra line break for better readability

        return reward


def append_to_file(object, filename = "game_log.txt"):
    """Function that appends an object to a file
    
    Params:
    ======
        object: The object to append
        filename: The name of the file to append to
    
    Returns:
    ======
        None"""
    with open(filename, "a") as f:
        f.write(str(object) + "\n")


if __name__ == "__main__":
    othello_env = OthelloEnv(num_players=1)
    othello_env.reset()


    while not othello_env.done:
        othello_env.render()
        legal_moves_list = othello_env.legal_moves()
        print("current player: " + str(othello_env.current_player)
                + " turn number: " + str(othello_env.turn_number))
        print("legal moves: " + str(legal_moves_list))
        #get the action from the user
        action = input("Enter the action: ")
        action = action.split(",")
        action = (int(action[0]), int(action[1]))
        
        while action not in legal_moves_list:
            print("Invalid action")
            print("legal moves: " + str(legal_moves_list))
            action = input("Enter the action: ")
            action = action.split(",")
            action = (int(action[0]), int(action[1]))

        print("action: " + str(action))
        othello_env.step(action)

    

