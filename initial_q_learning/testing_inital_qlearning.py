import numpy as np
from itertools import product

def generate_all_board_states():
    # Define the possible values for each location
    possible_values = [-1, 0, 1]

    # Generate all possible combinations for an 8x8 board
    all_combinations = product(possible_values, repeat=64)

    # Reshape each combination to represent an 8x8 board
    all_boards = np.array(list(all_combinations)).reshape(-1, 8, 8)

    # Convert the boards to strings for easy comparison and storage
    all_boards_as_strings = [board.tostring() for board in all_boards]

    return all_boards_as_strings

# Example usage
all_states = generate_all_board_states()
print("Number of possible states:", len(all_states))
