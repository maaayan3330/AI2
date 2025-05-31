import pressure_plate
import numpy as np

id = ["212437453"]


class Controller:
    """This class is a controller for a pressure plate game."""

    def __init__(self, game: pressure_plate.Game):
        """Initialize controller for given game model.
        """
        self.original_game = game

        # keep the all data i need for the MDP - that will help me work as fast as i can
        state = game.get_current_state()
        self.map = state[0]
        self.agent_pos = state[1]


        # init the model
        self.model = game.get_model()
        self.chosen_action_prob = self.model['chosen_action_prob']
        self.finished_reward = self.model['finished_reward']
        self.opening_door_reward = self.model['opening_door_reward']
        self.step_punishment = self.model['step_punishment']


    



    def choose_next_action(self, state):
        """Choose next action for a pressure plate game given the current state of the game.
        """
        return np.random.choice(["U", "R", "D", "L"])

