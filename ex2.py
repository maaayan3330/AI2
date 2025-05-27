import pressure_plate
import numpy as np

id = ["000000000"]


class Controller:
    """This class is a controller for a pressure plate game."""

    def __init__(self, game: pressure_plate.Game):
        """Initialize controller for given game model.
        """
        self.original_game = game

    def choose_next_action(self, state):
        """Choose next action for a pressure plate game given the current state of the game.
        """
        return np.random.choice(["U", "R", "D", "L"])

