from collections import deque
import pressure_plate
from ex1 import PressurePlateProblem
from search import astar_search
import numpy as np
import copy

# I used ChatGPT solely for syntax clarification, logic refinement, and function polishing. 
# the code and algorithmic decisions were written by me.
# During the assignment, I also shared a brainstorming session with Amit Bruhim and consulted with Tom Sasson.


id = ["212437453"]
#####################
AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3
WALL = 99
FLOOR = 98
#####################
DISCOUNT = 0.9
EPOCHS = 15
MAX_DEPTH = 4
WALL_LIMIT_S = 30
WALL_LIMIT_L = 40
MAKE_KEY = 20
GOOD_REWARD = 100
BAD_REWARD = 50
####################
ACTIONS = ["U", "L", "R", "D"]

DETERMINSITIC = {
                        'U': [1, 0, 0, 0],
                        'L': [0, 1, 0, 0],
                        'R': [0, 0, 1, 0],
                        'D': [0, 0, 0, 1]
        }

DIRECTION = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1) 
    }

class Controller:
    """
    Controller that strictly forces the agent to follow the A* path,
    step-by-step, regardless of stochastic results.
    """

    def __init__(self, game: pressure_plate.Game):
        """Initialize controller for given game model."""
        self.original_game = game
        self.map, self.agent_pos, _, _, _ = self.original_game.get_current_state()
        # Compute A* path using helper method and correct directions
        self.astar_path = self.compute_Astar_path()
        self.risky_keys_cache = {}
        # Create deterministic game state sequence along A* path
        self.a_star_game_states = self.create_game_object_by_action(self.astar_path)
        # Create policy and value tables based on A* path
        self.policy_sol = self.create_policy_valueTable_for_Astar(self.astar_path, self.a_star_game_states)
        # Extract goal location from map
        pos = np.argwhere(self.map == pressure_plate.GOAL)
        self.goal = (pos[0][0], pos[0][1])
        # Expand reachable states from A* path
        childs_states = self.collect_childs_states(self.a_star_game_states)
        # Run value iteration on reachable states
        self.V, self.policy = self.value_iteration(childs_states)

    # In this func - I restore the path of A* : My goal is to force the agent to walk in this path
    def compute_Astar_path(self):
        current_map, _, _, _, _ = self.original_game.get_current_state()
        problem = PressurePlateProblem(current_map)
        result = astar_search(problem)
        if result is None:
            return []
        result_node , _ = result
        path_nodes = list(reversed(result_node.path()))
        original_actions = [node.action for node in path_nodes if node.action is not None]
        corrected_actions = [self.correct_direction(a) for a in original_actions]
        return corrected_actions
    
    # In this func - I change the directions to match the check of the assignment
    def correct_direction(self, action):
        if action == 'U':
            return 'D'
        elif action == 'D':
            return 'U'
        return action

    # In this func - I want to create the all GAME objects , action by action in a deterministic environment
    # The goal of this is that i will know after every step how the board look like - and I will be able to suck up all the information
    def create_game_object_by_action(self, a_star_path):
        forced_games = []
        # becuase I dont want to destroy the original game
        current_game = copy.deepcopy(self.original_game)
        forced_games.append(current_game)
        # now i will run the all action like thay suppose to be 
        # Force the action by setting its probability to 100% for itself
        current_game._chosen_action_prob = DETERMINSITIC
        for action in a_star_path:
            # Submit action (now forced)
            current_game.submit_next_action(action)
            # Save the game copy after the move
            forced_games.append(copy.deepcopy(current_game))
        self.usefull_key = self.check_keys_in_Astar(forced_games[-1])
        return forced_games

    def check_keys_in_Astar(self, game):
        end = game.get_current_state()[0]
        opened = []
        for row in end:
            for cell in row:
                if WALL_LIMIT_S <= cell < WALL_LIMIT_L:
                    opened.append(cell - MAKE_KEY)
        return opened

    # In this func - I want to make V and policy based on the A* path : the point is that I know allreday what is the best policy so I build it
    # the goal - is that I can know what I want to append in my game and try to force my agent    
    def create_policy_valueTable_for_Astar(self , astar_path, a_star_game_states):
        # create v table of values + policy
        policy_for_Astar = {}   # policy(map) = action
        # i want to go all over the games - give more reward in up order for the states
        total_steps = len(a_star_game_states)

        for idx in range(total_steps):
            game = a_star_game_states[idx]
            game_map, _, _, _, _ = game.get_current_state()
            state_key = tuple(game_map.flatten())

            # Assign max value to final state, else assign descending values
            if idx == total_steps - 1:
                policy_for_Astar[state_key] = 'U'  # no action needed
            else:
                policy_for_Astar[state_key] = astar_path[idx]

        return policy_for_Astar

    # In this func - I want to do bfs and create 3 childs to every step in the A* path
    # the goal - to keep track if the agent split out side
    def collect_childs_states(self, base_games):
        child_nodes = []
        for base_game in base_games:
            seen_maps = set()
            queue = deque([(copy.deepcopy(base_game), 0)])
            while queue:
                game_state, depth = queue.popleft()
                # Don't go deeper than max allowed depth
                if depth >= MAX_DEPTH:
                    break
                # Hash the current map to avoid revisits
                map_arr = game_state.get_current_state()[0]
                state_hash = tuple(map_arr.flatten())
                if state_hash in seen_maps:
                    continue
                seen_maps.add(state_hash)
                # Store the reachable game state
                child_nodes.append(copy.deepcopy(game_state))
                # Expand possible next steps
                for action in ['U', 'L', 'R', 'D']:
                    next_game = copy.deepcopy(game_state)
                    # Force deterministic behavior
                    next_game._chosen_action_prob = DETERMINSITIC
                    # Apply the action and enqueue
                    next_game.submit_next_action(action)
                    queue.append((next_game, depth + 1))

        return child_nodes

    # ----------------------- VI - functions -----------------------------
    def value_iteration(self, reachable_states):
        V = {}
        pai = {}
        reachable = [game.get_current_state() for game in reachable_states]

        for state in reachable:
            key = tuple(state[0].flatten())
            V[key] = self.init_v_table(state)
            pai[key] = 'U'

        for _ in range(EPOCHS):
            new_V = {}
            for state in reachable:
                key = tuple(state[0].flatten())
                best_val, best_act = self.evaluate_actions(state, V)
                new_V[key] = best_val
                pai[key] = best_act
            V = new_V

        return V, pai

    def evaluate_actions(self, state, V):
        best_val = -np.inf
        best_act = 'U'
        for action in ACTIONS:
            expected_val = 0.0
            base = self.restore_game_state(state)

            for i, _ in enumerate(ACTIONS):
                p = base._chosen_action_prob[action][i]
                sim = copy.deepcopy(base)
                sim._chosen_action_prob = DETERMINSITIC
                sim.submit_next_action(action)
                next_state = sim.get_current_state()
                h_next = tuple(next_state[0].flatten())
                reward = sim.get_current_reward() + self.add_reward(next_state)
                expected_val += p * (reward + DISCOUNT * V.get(h_next, 0))

            if expected_val > best_val:
                best_val = expected_val
                best_act = action

        return best_val, best_act
    
    def restore_game_state(self, state):
        game = copy.deepcopy(self.original_game)
        game._map = state[0].copy()
        game._agent_pos = state[1]
        game._steps = state[2]
        game._done = state[3]
        game._successful = state[4]
        return game

    # ----------------------- VI - functions -----------------------------
    
    # this func is to try make the agent even more to not get out from the path
    def add_reward(self, state):
        R = 0
        hashable_state = tuple(state[0].flatten())
        map = state[0]
        # if it a good step in A* -> add reward
        if hashable_state in self.policy_sol:
            R += GOOD_REWARD
        # if it is a key that is usefull -> check we didnt stack him 
        for key in self.usefull_key:
            R += self.penalty_if_blocked(map, key)
        # check also if it a door that i wont open in A* so i dont want to get near her
        for door_number in range(40, 50):
            corresponding_key = door_number - 30
            if corresponding_key not in self.usefull_key:
                R += self.penalty_if_blocked(map, door_number, penalty=BAD_REWARD // 2)
        return R

    # here it is the all check with the corner thing
    def penalty_if_blocked(self, map, key_s, penalty=BAD_REWARD):
        key = (key_s, map.shape)
        if key in self.risky_keys_cache:
            return -penalty if self.risky_keys_cache[key] else 0

        positions = np.argwhere(map == key_s)
        for i, j in positions:
            blocked = 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if not (0 <= ni < map.shape[0] and 0 <= nj < map.shape[1]):
                    continue
                neighbor = map[ni, nj]
                # i check if it a wall or!! - a door that not supposed to open
                if neighbor == WALL or (40 <= neighbor < 50 and (neighbor - 30) not in self.usefull_key):
                    blocked += 1
                if blocked >= 2:
                    self.risky_keys_cache[key] = True
                    return -penalty

        self.risky_keys_cache[key] = False
        return 0

    def init_v_table(self, state):
        agent_pos = state[1]
        return -abs(agent_pos[0] - self.goal[0]) - abs(agent_pos[1] - self.goal[1])
    

    def choose_next_action(self, state):
        h_state = tuple(state[0].flatten())

        # Follow A* path if still on it
        if h_state in self.policy_sol:
            return self.policy_sol[h_state]

        # Use value iteration policy if reachable
        if h_state in self.policy:
            return self.policy[h_state]

        # If completely off-path, reroute using A* to goal
        current_map = state[0]
        reroute_problem = PressurePlateProblem(current_map)
        reroute_result = astar_search(reroute_problem)

        if reroute_result is not None:
            reroute_node, _ = reroute_result
            path_nodes = list(reversed(reroute_node.path()))
            reroute_actions = [node.action for node in path_nodes if node.action is not None]
            corrected_actions = [self.correct_direction(a) for a in reroute_actions]

            if corrected_actions:
                return corrected_actions[0]

        # If no path found, move randomly
        return np.random.choice(["U", "R", "D", "L"])