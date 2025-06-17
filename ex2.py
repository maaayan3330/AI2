from collections import deque
import pressure_plate
from ex1 import PressurePlateProblem
from search import astar_search
import numpy as np
import copy

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

DETERMINSITIC_PROBABLITIES = {
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
        self.solution_exist = len(self.astar_path) > 0
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
        current_game._chosen_action_prob = DETERMINSITIC_PROBABLITIES
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
                    next_game._chosen_action_prob = DETERMINSITIC_PROBABLITIES
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
                sim._chosen_action_prob = DETERMINSITIC_PROBABLITIES
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

    def add_reward(self, state):
        R = 0
        hashable_state = tuple(state[0].flatten())
        if hashable_state in self.policy_sol:
            R += GOOD_REWARD
        map = state[0]
        for door_number in self.usefull_key:
            positions = np.argwhere(map == door_number)
            for i, j in positions:
                walls = 0
                if i > 0 and map[i-1, j] == WALL: walls += 1
                if i < map.shape[0]-1 and map[i+1, j] == WALL: walls += 1
                if j > 0 and map[i, j-1] == WALL: walls += 1
                if j < map.shape[1]-1 and map[i, j+1] == WALL: walls += 1
                if walls >= 2:
                    R -= BAD_REWARD
        return R

    def init_v_table(self, state):
        agent_pos = state[1]
        return -abs(agent_pos[0] - self.goal[0]) - abs(agent_pos[1] - self.goal[1])

    
    # def choose_next_action(self, state):
    #     h_state = tuple(state[0].flatten())

    #     if h_state in self.policy_sol:
    #         return self.policy_sol[h_state]

    #     if h_state in self.policy:
    #         return self.policy[h_state]

    #     if not self.solution_exist:
    #         return self.recover_towards_astar(state)

    #     # נסה לחשב מסלול חדש לגמרי
    #     current_map = state[0]
    #     problem = PressurePlateProblem(current_map)
    #     result = astar_search(problem)
    #     if result is None:
    #         self.solution_exist = False
    #         return self.recover_towards_astar(state)

    #     self.solution_exist = True
    #     result_node, _ = result
    #     path_nodes = list(reversed(result_node.path()))
    #     original_actions = [node.action for node in path_nodes if node.action is not None]
    #     corrected_actions = [self.correct_direction(a) for a in original_actions]
    #     sol_nodes = self.create_game_object_by_action(corrected_actions)
    #     self.policy_solution = self.create_policy_valueTable_for_Astar(corrected_actions, sol_nodes)

    #     if h_state in self.policy_solution:
    #         return self.policy_solution[h_state]
    #     if h_state in self.policy:
    #         return self.policy[h_state]

    #     return self.recover_towards_astar(state)

    # def recover_towards_astar(self, state):
    #     current_map, agent_pos, *_ = state

    #     # חפש את המצב הכי קרוב במסלול A*
    #     best_dist = float('inf')
    #     best_target_map = None

    #     for game in self.a_star_game_states:
    #         map_astar = game.get_current_state()[0]
    #         pos_astar = np.argwhere(map_astar == AGENT)
    #         if len(pos_astar) == 0:
    #             continue
    #         pos_astar = tuple(pos_astar[0])
    #         dist = abs(agent_pos[0] - pos_astar[0]) + abs(agent_pos[1] - pos_astar[1])
    #         if dist < best_dist:
    #             best_dist = dist
    #             best_target_map = map_astar

    #     if best_target_map is None:
    #         return np.random.choice(ACTIONS)

    #     # חפש פעולה לפי A* למצב הקרוב
    #     problem = PressurePlateProblem(current_map)
    #     result = astar_search(problem)

    #     if result is None:
    #         return np.random.choice(ACTIONS)

    #     result_node, _ = result
    #     path_nodes = list(reversed(result_node.path()))
    #     for node in path_nodes:
    #         if node.action is not None:
    #             return self.correct_direction(node.action)

    #     return np.random.choice(ACTIONS)

    def choose_next_action(self, state):
        h_state = tuple(state[0].flatten())

        if h_state in self.policy_sol:
            return self.policy_sol[h_state]

       
        if h_state in self.policy:
            return self.policy[h_state]

        current_pos = state[1] 

        min_distance = float('inf')
        closest_state = None
        closest_index = None

        for idx, game in enumerate(self.a_star_game_states):
            _, agent_pos, _, _, _ = game.get_current_state()
            dist = abs(agent_pos[0] - current_pos[0]) + abs(agent_pos[1] - current_pos[1])
            if dist < min_distance:
                min_distance = dist
                closest_state = game
                closest_index = idx

        if closest_state is not None:
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

        return np.random.choice(["U", "R", "D", "L"])
