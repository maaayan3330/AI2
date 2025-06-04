from pprint import pprint
import numpy as np

AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3
WALL = 99
FLOOR = 98


class Game:
    """Game class --- presents a pressure plate game played for given number of steps."""

    def __init__(self, max_steps, i_map, model, debug=False):
        """Initialize the Game class.
        max_steps - represents the number of steps the game is run
        map - the initial state of the map
        model - the model probabilities and parameters"""
        self._max_steps = max_steps
        self._map = np.array(i_map)
        self._model = model
        self._debug = debug
        if self._debug:
            self._history = list()
        self._steps = 0
        self._reward = 0
        self._chosen_action_prob = model['chosen_action_prob']
        self._finished_reward = model['finished_reward']
        self._opening_door_reward = model['opening_door_reward']
        self._step_punishment = model['step_punishment']
        self._seed = model['seed']
        self._agent_pos = tuple(np.argwhere(self._map == AGENT).flatten())
        self._done = False
        self._successful = False
        np.random.seed(self._seed)

    def get_current_state(self):
        """
        Tuple of the current state of the game.
        :return: numpy array (current state of map), tuple (agent position),
        scalar (steps passed), boolean (is run finished), boolean (is run successful)
        """

        return self._map.copy(), self._agent_pos, self._steps, self._done, self._successful

    def get_max_steps(self):
        """
        Tuple of the current state of the game.
        :return: numpy array (current state of map), tuple (agent position),
        scalar (steps passed), boolean (is run finished)
        """

        return self._max_steps

    def get_current_reward(self):
        """
        Current reward of the game
        :return: scalar
        """
        return self._reward

    def get_model(self):
        """
        Dictionary of detailing the model of the game.
        :return: dictionary (detailed in the pdf)
        """
        return self._model

    def _open_doors(self, key):
        num_plates = np.argwhere(self._map == key + 10).size
        if num_plates == 0:
            doors = self._map == key + 30
            self._map[doors] = FLOOR
            return self._opening_door_reward[key]
        return 0

    def submit_next_action(self, chosen_action):
        """
        Takes chosen action from user and updates the game from its consequences.
        :param chosen_action: Letter from the set {'U', 'L', 'R', 'D'}
        """
        if self._done:
            return
        action = np.random.choice(["U", "L", "R", "D"], p=self._chosen_action_prob[chosen_action[0].upper()])
        if self._debug:
            submit_result = list()
            submit_result.append(f'step {self._steps}, chosen action: {chosen_action[0].upper()}, action: {action}')
            self._history.append(submit_result)
        add_reward = 0
        i = self._agent_pos[0]
        j = self._agent_pos[1]
        next_i = i
        next_j = j
        next_next_i = i
        next_next_j = j

        if action == 'U':
            next_i = min(i + 1, self._map.shape[0] - 1)
            next_next_i = min(i + 2, self._map.shape[0] - 1)
        elif action == 'D':
            next_i = max(i - 1, 0)
            next_next_i = max(i - 2, 0)
        elif action == 'R':
            next_j = min(j + 1, self._map.shape[1] - 1)
            next_next_j = min(j + 2, self._map.shape[1] - 1)
        elif action == 'L':
            next_j = max(j - 1, 0)
            next_next_j = max(j - 2, 0)

        if self._map[next_i, next_j] == FLOOR or self._map[next_i, next_j] == GOAL:
            if self._map[next_i, next_j] == FLOOR:
                self._map[next_i, next_j] = AGENT
            elif self._map[next_i, next_j] == GOAL:
                add_reward = self._finished_reward
                self._map[next_i, next_j] = AGENT_ON_GOAL
                self._done = True
                self._successful = True
            self._agent_pos = (next_i, next_j)
            self._map[i, j] = FLOOR
        elif 10 <= self._map[next_i, next_j] <= 19 and (self._map[next_next_i, next_next_j] == FLOOR or
                                                        self._map[next_next_i, next_next_j] == 10 +
                                                        self._map[next_i, next_j]):
            if self._map[next_next_i, next_next_j] != FLOOR:
                self._map[next_next_i, next_next_j] = self._map[next_i, next_j] + 20
                add_reward = self._open_doors(self._map[next_i, next_j])
            else:
                self._map[next_next_i, next_next_j] = self._map[next_i, next_j]
            self._map[next_i, next_j] = AGENT
            self._agent_pos = (next_i, next_j)
            self._map[i, j] = FLOOR

        self._reward += add_reward
        self._steps += 1

        if self._steps == self._max_steps:
            self._done = True
        if self._done:
            self._reward += self._steps * self._step_punishment

    def show_history(self):
        """
        Debug function used to see the probabilities and the process of the game.
        """
        if self._debug:
            print('History:')
            pprint(self._history)


def create_pressure_plate_game(game):
    if game[3]:
        print('--------DEBUG MODE--------')
        print('<< create pressure plate game >>')
        print('<maximize R on>', game[1])
        print('in', game[0], 'steps')
        print('under these conditions:')
        pprint(game[2])
    return Game(*game)


example = {'chosen_action_prob': {'U': [0.9, 0.33, 0.33, 0.34], 'L': [0.8, 0.20, 0.40, 0.40],
              'R': [0.7, 0.50, 0.10, 0.40], 'D': [0.6, 0.80, 0.15, 0.05]},
           'finished_reward': 150,
           'opening_door_reward': {10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 17: 8, 18: 9, 19: 10},
           'step_punishment': -1,
           'seed': 42}
# print("up".upper())

