from collections import deque
import pressure_plate
import numpy as np
from ex1 import PressurePlateProblem
from search import astar_search
import copy

id = ["209379239"]

BLANK = 0
WALL = 99
FLOOR = 98
AGENT = 1
GOAL = 2
LOCKED_DOORS = list(range(40, 50))
PRESSED_PLATES = list(range(30, 40))
PRESSURE_PLATES = list(range(20, 30))
KEY_BLOCKS = list(range(10, 20))
ACTION_REWARD = -2
ITERATIONS = 15
DISCOUNT_FACTOR = 0.9
DEPTH = 4

ACTIONS = ["U", "L", "R", "D"]

ACTION_OFFSETS = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0,1)
        }

DETERMINSITIC_PROBABLITIES = {
                        'U': [1, 0, 0, 0],
                        'L': [0, 1, 0, 0],
                        'R': [0, 0, 1, 0],
                        'D': [0, 0, 0, 1]
        }

class Controller:
    """This class is a controller for a pressure plate game."""
    
    def __init__(self, game: pressure_plate.Game):
        """Initialize controller for given game model.
        """
        # initialize class attributes
        self.original_game = game
        self.map, self.agent_pos, _, _, _ = self.original_game.get_current_state()
        self.row_len, self.col_len = len(self.map), len(self.map[0])
        self.goal_reward = self.original_game._finished_reward
        self.A_star_path = self.get_A_star_path(self.map) # list of optimal path actions computed deterministicly with U and D replaced
        self.optimal_path_nodes = self.get_optimal_path_nodes() # list of game nodes of the optimal path
        self.optimal_path_policy = self.get_optimal_path_policy() # use this policy if on optimal path 
        self.goal_pos =  self.get_goal_pos()
        self.children_of_optimal_path = self.get_children_of_optimal_path() # list of game nodes including the optimal path and their children until a fixed depth
        self.policy_children = self.get_children_of_optimal_path_policy()
        
    def get_A_star_path(self, map):
        """Computes A* path using ex1 functions, removes last element and swaps U and D"""
        # use A* algorithm
        game = PressurePlateProblem(map)
        output = astar_search(game)
        if output is None:
            print("no optimal path found")
            return []
        result, _ = output
        
        # revese the list and remove last element
        path = list(reversed(result.path()))
        solution =  [node.action for node in path if node.action is not None]

        # swap U and D to fit implementaion of pressure plate class
        swap = {'U': 'D', 'D': 'U'}
        replaced_solution = [swap.get(a, a) for a in solution]
        return replaced_solution

    def get_optimal_path_nodes(self):
        """Create a list of game objects for each step in the path"""
        # create copy of the game object and append to list
        game_copy = copy.deepcopy(self.original_game)
        optimal_path_nodes = []
        optimal_path_nodes.append(game_copy)

        # force deterministic environmet
        game_copy._chosen_action_prob = DETERMINSITIC_PROBABLITIES

        # iterate through the actions and append game objects to the list
        # after the action was done on them
        for action in self.A_star_path:
            game_copy.submit_next_action(action)
            optimal_path_nodes.append(copy.deepcopy(game_copy))

        return optimal_path_nodes

    def get_optimal_path_policy(self): 
        """Create the optimal's path values and policy"""
        # initialize dictionary
        optimal_path_policy = {}

        # iterate through optimal_path_nodes 
        for i, game in enumerate(self.optimal_path_nodes[:-1]): # excluding the last state - goal
            game_state = game.get_current_state()
            hashed_game_map = tuple(game_state[0].flatten())
            optimal_path_policy[hashed_game_map] = self.A_star_path[i]

        # handle the last game - goal
        last_game = self.optimal_path_nodes[-1]
        last_game_state = last_game.get_current_state()
        hashed_game_map = tuple(last_game_state[0].flatten())

        # add to policy
        optimal_path_policy[hashed_game_map] = 'U'

        return optimal_path_policy

    def get_goal_pos(self):
        """Return position of the goal in the map"""
        position = np.argwhere(self.map == pressure_plate.GOAL)
        return ((int(position[0][0]), int(position[0][1])))

    def get_children_of_optimal_path(self):
        """Iterates on all the optimal path nodes, and creates for each node a tree - 
        game maps for every possible action until certain depth"""
        children_of_optimal_path = []
        for game in self.optimal_path_nodes:
            # initialize BFS algorithm
            visited = set()
            queue = deque()
            queue.append((copy.deepcopy(game), 0)) # start iterating from the optimal node
            while queue:
                # pop queue and check if reached max depth
                game_node, depth = queue.popleft()
                if depth >= DEPTH:
                    break

                # get the current node map
                game_state = game_node.get_current_state()
                game_map = game_state[0]
                hashed_map = tuple(game_map.flatten())

                # check if already reached this node - if not, add to set and list
                if hashed_map in visited:
                    continue
                visited.add(hashed_map)
                children_of_optimal_path.append(game_node)

                # develope the children of the current game node
                for action in ACTIONS:
                    next_game_node = copy.deepcopy(game_node)
                    next_game_node._chosen_action_prob = DETERMINSITIC_PROBABLITIES
                    next_game_node.submit_next_action(action)
                    queue.append((next_game_node, depth + 1))

        return children_of_optimal_path

    def get_children_of_optimal_path_policy(self):
        """Perform value iteration for the children of optimal path and return the values and policy"""
        V = {}
        policy = {}
        optimal_path_states = [game.get_current_state() for game in self.children_of_optimal_path]

        # initialize V and policy
        for state in optimal_path_states:
            hashed = tuple(state[0].flatten())
            V[hashed] = 0
            policy[hashed] = 'L'

        # perform value iteration
        for _ in range(ITERATIONS):
            new_V = {}
            for state in optimal_path_states:
                hashed = tuple(state[0].flatten())
                best_value, best_action = self.perform_value_iteration_step(state, V)
                new_V[hashed] = best_value
                policy[hashed] = best_action
            V = new_V

        return policy

    def perform_value_iteration_step(self, state, V):
        """Compute the best value and action for a single state during value iteration"""
        best_value = -np.inf
        best_action = 'U'

        for action in ACTIONS:
            expectation = 0.0
            map, agent_pos, steps, done, successful = state
            base_game = copy.deepcopy(self.original_game)
            base_game._map = map.copy()
            base_game._agent_pos = agent_pos
            base_game._steps = steps
            base_game._done = done
            base_game._successful = successful

            for i, _ in enumerate(ACTIONS):
                probability = base_game._chosen_action_prob[action][i]
                simulated_game = copy.deepcopy(base_game)
                simulated_game._chosen_action_prob = DETERMINSITIC_PROBABLITIES
                simulated_game.submit_next_action(action)
                next_state = simulated_game.get_current_state()
                hashable_next = tuple(next_state[0].flatten())
                V_next = V.get(hashable_next, 0)
                reward = simulated_game.get_current_reward()
                expectation += probability * (reward + DISCOUNT_FACTOR * V_next)

            if expectation > best_value:
                best_value = expectation
                best_action = action

        return best_value, best_action

    # def choose_next_action(self, state):
    #     """Choose next action for a pressure plate game given the current state of the game.
    #     """
    #     # hash the current map
    #     hashed_game_map = tuple(state[0].flatten())

    #     # if on optimal path - use its policy
    #     if hashed_game_map in self.optimal_path_policy:
    #         return self.optimal_path_policy[hashed_game_map]
        
    #     # if on one of the children of the optimal path - until DEPTH 0 - use its policy 
    #     if hashed_game_map in self.policy_children:
    #         return self.policy_children[hashed_game_map]
        
    #     # if none of those, choose randomly
    #     return np.random.choice(["U", "R", "D", "L"])

    def correct_direction(self, action):
        if action == 'U':
            return 'D'
        elif action == 'D':
            return 'U'
        return action
    
    def choose_next_action(self, state):
        hashed_game_map = tuple(state[0].flatten())

        if hashed_game_map in self.optimal_path_policy:
            return self.optimal_path_policy[hashed_game_map]

       
        # if on one of the children of the optimal path - until DEPTH 0 - use its policy 
        if hashed_game_map in self.policy_children:
            return self.policy_children[hashed_game_map]

        current_pos = state[1] 

        min_distance = float('inf')
        closest_state = None
        closest_index = None

        for idx, game in enumerate(self.optimal_path_nodes):
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