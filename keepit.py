import pressure_plate
import numpy as np
import copy

id = ["212437453"]

AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3
WALL = 99
FLOOR = 98
MAX_DEPTH = 3
DISCOUNT = 0.9

class Controller:
    """This class is a controller for a pressure plate game."""

    def __init__(self, game: pressure_plate.Game):
        """Initialize controller for given game model.
        """
        self.original_game = game

        # keep the all data i need for the MDP - that will help me work as fast as i can
        state = game.get_current_state()
        self.max_steps = game.get_max_steps()
        self.map = state[0]
        self.agent_pos = state[1]
        rows, cols = self.map.shape

        # init the model
        self.model = game.get_model()
        self.chosen_action_prob = self.model['chosen_action_prob']
        self.finished_reward = self.model['finished_reward']
        self.opening_door_reward = self.model['opening_door_reward']
        self.step_punishment = self.model['step_punishment']

        self.max_depth = MAX_DEPTH
        self.discount = DISCOUNT


        # init all V0(s)
        # helper to freeze a map so we can use it as a key
        def freeze_map(map_array):
            return tuple(map_array.flatten())
        self.freeze_map = freeze_map

        # value function: (frozen_map, steps_left) -> value
        self.V = {}

        # # i will init every state in live - if it allready exiset i will get it otherwise i will make it to 0 - Dynamic Programming with memoization
        def get_value(self, frozen_map, steps_left):
            key = (frozen_map, steps_left)
            if key not in self.V:
                self.V[key] = 0
            return self.V[key]


        self.get_value = get_value

                
    # this function is to make the agent want to get closer to - 1) key blockes, 2) goal
    # becuase -2 all the time it is not mach
    def heuristic_value(self, state):
        game_map, agent_pos, steps_taken, done, success = state
        reward = 0

        # קרבה למטרה
        goal_positions = np.argwhere((game_map == GOAL) | (game_map == AGENT_ON_GOAL))
        if len(goal_positions) > 0:
            goal_pos = goal_positions[0]
            reward += 5 / (1 + np.abs(agent_pos[0] - goal_pos[0]) + np.abs(agent_pos[1] - goal_pos[1]))  # ככל שקרוב, יותר טוב

        # קרבה ללוח (מפתחות) - ערכים 10 עד 19
        plate_positions = np.argwhere((game_map >= 10) & (game_map <= 19))
        if len(plate_positions) > 0:
            dists = [np.abs(agent_pos[0] - p[0]) + np.abs(agent_pos[1] - p[1]) for p in plate_positions]
            reward += 3 / (1 + min(dists))  # ככל שקרוב ללוח, יותר טוב

        # עונש על קרבה לדלת סגורה (30–39)
        # door_positions = np.argwhere((game_map >= 30) & (game_map <= 39))
        # if len(door_positions) > 0:
        #     dists = [np.abs(agent_pos[0] - p[0]) + np.abs(agent_pos[1] - p[1]) for p in door_positions]
        #     reward -= 2 / (1 + min(dists))  # להרתיע מלהיתקע בדלת

        return reward

    # this fucntion will get a state that exiest -> make a copy of the game in this state
    # use the function submit_next_action on the copy -> return the new state and the reward this action gave
    def simulate_action(self, state, action):
        """
        Simulates the result of taking `action` from `state`,
        using deepcopy of the original game and submit_next_action.

        Returns:
            next_state: the resulting state tuple (map, pos, steps, done, success)
            reward: reward gained from this action
        """
        # שלוף נתונים מהמצב הנוכחי
        current_map, current_pos, steps_so_far, _, _ = state

        # צור עותק עמוק של המשחק
        sim_game = copy.deepcopy(self.original_game)

        # עדכן את עותק המשחק למצב הרצוי
        sim_game._map = np.copy(current_map)
        sim_game._agent_pos = current_pos
        sim_game._steps = steps_so_far
        sim_game._reward = 0  # נחשב רק את התגמול של הצעד הזה

        # בצע את הפעולה
        sim_game.submit_next_action(action)

        # קבל את המצב הבא ואת התגמול שהתקבל
        next_state = sim_game.get_current_state()
        reward = sim_game.get_current_reward()

        return next_state, reward
    


    # in this function i reurn the new val : Vk+1(s) <- max ∑ P(s, a, s')[R(s, a, s') + discount*Vk(s')]
    # - i need to check - R U D L
    # inthis function i get - the map of a old state , how much steps left, take an action
    def expected_value(self, state, depth, chosen_action):
        """
        Computes the expected value of performing chosen_action at the given state,
        by simulating all possible resulting actions (due to stochasticity).

        Arguments:
            state: tuple (map, agent_pos, steps_taken, done, success)
            depth: current depth in recursion (0..MAX_DEPTH)
            chosen_action: str, one of "U", "D", "L", "R"

        Returns:
            expected_value: float
        """
        total_value = 0
        # רשימת כל הפעולות האפשריות בפועל (U/L/R/D)
        actual_actions = ["U", "L", "R", "D"]
        probs = self.chosen_action_prob[chosen_action]

        for idx, actual_action in enumerate(actual_actions):
            prob = probs[idx]
            next_state, reward = self.simulate_action(state, actual_action)

            if depth == self.max_depth or next_state[3]:  # next_state[3] == done
                value = reward + self.heuristic_value(next_state)
            else:
                value = reward + self.discount * self.best_action_value(next_state, depth + 1)

            total_value += prob * value

        return total_value


    def best_action_value(self, state, depth):
        """
        Returns the maximum expected value achievable from this state
        by choosing the best possible action at current depth.
        """
        best_val = -float('inf')
        for action in ["U", "D", "L", "R"]:
            val = self.expected_value(state, depth, action)
            if val > best_val:
                best_val = val
        return best_val

       

    def choose_next_action(self, state):
        """Choose next action for a pressure plate game given the current state of the game.
        """
        # print(">> Choosing next action...")
        best_val = -float('inf')
        best_action = "U"  # ברירת מחדל

        for action in ["U", "D", "L", "R"]:
            val = self.expected_value(state, 0, action)
            # print(f"[DEPTH=0] Trying action: {action} -> expected value: {val}")

            if val > best_val:
                best_val = val
                best_action = action
        # print(f"Chosen action: {best_action}")
        return best_action

