from collections import deque
import pressure_plate
from ex1 import PressurePlateProblem
from search import astar_search
import numpy as np
import copy

id = ["212437453"]

AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3
WALL = 99
FLOOR = 98


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
        self.original_game = game
        self.map, self.agent_pos, _, _, _ = self.original_game.get_current_state()
        # Get the opt path from A*
        self.astar_path = self.compute_Astar_path()
        print("🧭 Forced A* path:", self.astar_path)
        # get the obejects to any step in the A* path - extra check in case it emmpty
        if self.astar_path:
            self.a_star_game_states = self.create_game_object_by_action(self.astar_path)
        else:
            self.a_star_game_states = []


    # In this func - I restore the path of A* : My goal is to force the agent to walk in this path
    def compute_Astar_path(self):
        current_map, _, _, _, _ = self.original_game.get_current_state()
        problem = PressurePlateProblem(current_map)
        result_node, _ = astar_search(problem)
        if result_node is None:
            return []
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
        # now i will run the all action like thay suppose to be 
        for action in a_star_path:
            # Force the action by setting its probability to 100% for itself
            original_probs = current_game._chosen_action_prob[action]
            current_game._chosen_action_prob[action] = [0, 0, 0, 0]
            idx = {'U': 0, 'L': 1, 'R': 2, 'D': 3}[action]
            current_game._chosen_action_prob[action][idx] = 1.0
            
            # Submit action (now forced)
            current_game.submit_next_action(action)

            # Save the game copy after the move
            forced_games.append(copy.deepcopy(current_game))

            # Restore original probabilities to avoid corrupting model
            current_game._chosen_action_prob[action] = original_probs

        return forced_games

    # In this func - I want to make V and policy based on the A* path : the point is that I know allreday what is the best policy so I build it
    # the goal - is that I can know what I want to append in my game and try to force my agent    
    def create_policy_valueTable_for_Astar(self , astar_path, a_star_game_states):
        # create v table of values + policy
        v_table_for_Astar = {}  # v(map) = int
        policy_for_Astar = {}   # policy(map) = action
        # i want to go all over the games - give more reward in up order for the states
        total_steps = len(a_star_game_states)

        for idx in range(total_steps):
            game = a_star_game_states[idx]
            game_map, _, _, _, _ = game.get_current_state()
            state_key = tuple(game_map.flatten())

            # Assign max value to final state, else assign descending values
            if idx == total_steps - 1:
                v_table_for_Astar[state_key] = total_steps
                policy_for_Astar[state_key] = 'U'  # no action needed
            else:
                v_table_for_Astar[state_key] = idx
                policy_for_Astar[state_key] = astar_path[idx]

        return v_table_for_Astar, policy_for_Astar

    # In this func - I want to do bfs and create 3 childs to every step in the A* path
    # the goal - to keep track if the agent split out side
    def collect_reachable_states(self, base_games, max_depth=3):
        child_nodes = []

        for base_game in base_games:
            seen_maps = set()
            queue = deque([(copy.deepcopy(base_game), 0)])

            while queue:
                game_state, depth = queue.popleft()

                # Don't go deeper than max allowed depth
                if depth > max_depth:
                    continue

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
                    idx = {'U': 0, 'L': 1, 'R': 2, 'D': 3}[action]
                    next_game._chosen_action_prob[action] = [0.0, 0.0, 0.0, 0.0]
                    next_game._chosen_action_prob[action][idx] = 1.0

                    # Apply the action and enqueue
                    next_game.submit_next_action(action)
                    queue.append((next_game, depth + 1))

        return child_nodes

    
   
    def choose_next_action(self, state):
        """Choose next action for a pressure plate game given the current state of the game.
        """
        return np.random.choice(["U", "R", "D", "L"])
   
