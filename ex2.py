import pressure_plate
import numpy as np
import copy

id = ["212437453"]

AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3
WALL = 99
FLOOR = 98
MAX_DEPTH = 6
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

        # init all V0(s)
        # helper to freeze a map so we can use it as a key
        def freeze_map(map_array):
            return tuple(map_array.flatten())
        self.freeze_map = freeze_map

        # value function: (frozen_map, steps_left) -> value
        self.V = {}

        # i will init every state in live - if it allready exiset i will get it otherwise i will make it to 0 - Dynamic Programming with memoization
        def get_value(state):
            if state not in self.V:
                self.V[state] = 0
            return self.V[state]

        self.get_value = get_value

                

    # this fucntion will get a state that exiest -> make a copy of the game in this state
    # use the function submit_next_action on the copy -> return the new state and the reward this action gave
    def simulate_action_on_copy(self, frozen_map, steps_left, action):
        # print(f"[DEBUG] simulate_action_on_copy: action={action}, steps_left={steps_left}")

        # 1 - first make a copy - it will be type game
        new_game = copy.deepcopy(self.original_game)


        # 2 - take the frozen map and remake it
        map_shape = self.map.shape
        map_array = np.array(frozen_map).reshape(map_shape)
        new_game._map = map_array

        # 3 - find the agent position - mabay the agent is on goal
        agent_locs = np.argwhere((new_game._map == AGENT) | (new_game._map == AGENT_ON_GOAL))
        new_game._agent_pos = agent_locs[0] if len(agent_locs) > 0 else (-1, -1)


        # now i can change the map safley - i will keep the reward before i do the action
        if hasattr(new_game, "_reward"):
            reward_before = new_game._reward
        else:
            reward_before = 0
        
        # do the action - 
        new_game.submit_next_action(action)
        # keep the new reward - submit is the one that updated the reward
        reward_after = new_game._reward
        reward = reward_after - reward_before

        frozen_next_map = self.freeze_map(new_game._map)
        next_steps_left = steps_left - 1
        # return the new state + how mach steps left
        return (frozen_next_map, next_steps_left), reward


    # in this function i reurn the new val : Vk+1(s) <- max ∑ P(s, a, s')[R(s, a, s') + discount*Vk(s')]
    # - i need to check - R U D L
    # inthis function i get - the map of a old state , how much steps left, take an action
    def expected_value(self, frozen_map, steps_left, chosen_action):
        expected = 0

        possible_actions = ["U", "L", "R", "D"]
        probs = self.chosen_action_prob[chosen_action]

        for actual_action, prob in zip(possible_actions, probs):
            if prob == 0:
                continue

            next_state, reward = self.simulate_action_on_copy(frozen_map, steps_left, actual_action)

            # נוסיף פה את העומק! נריץ value_iteration על המצב הבא
            if next_state[1] > 0:  # יש צעדים
                self.value_iteration(*next_state)

            value = self.get_value(next_state)

            expected += prob * (reward + DISCOUNT * value)
        # print(f"[DEBUG] expected_value: chosen_action={chosen_action}, steps_left={steps_left}")


        return expected



    def value_iteration(self, frozen_map, steps_left):
        print(f"\n[DEBUG] value_iteration: steps_left={steps_left}, real_depth={self.max_steps - steps_left}, state={frozen_map}")

         # חישוב עומק הרקורסיה לפי כמה צעדים כבר עברו
        real_depth = self.max_steps - steps_left
        if steps_left == 0 or real_depth >= MAX_DEPTH:
            return

        state = (frozen_map, steps_left)

        best_value = float('-inf')

        for action in ["U", "L", "R", "D"]:
            ev = self.expected_value(frozen_map, steps_left, action)
            best_value = max(best_value, ev)

        self.V[state] = best_value


    def choose_next_action(self, state):
        """Choose next action for a pressure plate game given the current state of the game.
        """

        map_array, agent_pos, steps_taken, done, success = state
        steps_left = self.max_steps - steps_taken - 1
        print(f"\n[DEBUG] choose_next_action: steps_left={steps_left}, agent_pos={agent_pos}")
        self.game = copy.deepcopy(self.original_game)
        self.game._map = np.copy(map_array)
        self.game._agent_pos = agent_pos

        if steps_left <= 0:
            return "U"  # ברירת מחדל בטוחה

        frozen_map = self.freeze_map(map_array)
        self.value_iteration(frozen_map, steps_left)

        best_action = None
        best_value = float('-inf')

        for action in ["U", "L", "R", "D"]:
            ev = self.expected_value(frozen_map, steps_left, action)
            print(f"action={action}, expected_value={ev}") 
            if ev > best_value:
                best_value = ev
                best_action = action

        return best_action if best_action is not None else "U"

