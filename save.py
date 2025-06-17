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

DISCOUNT = 0.9
EPOCHS = 15
MAX_DEPTH = 4
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
        # self.original_game = game
        # self.map, self.agent_pos, _, _, _ = self.original_game.get_current_state()
        # # Get the opt path from A*
        # self.astar_path = self.compute_Astar_path()
        # print("🧭 Forced A* path:", self.astar_path)
        # # get the obejects to any step in the A* path - extra check in case it emmpty
        # self.a_star_game_states = self.create_game_object_by_action(self.astar_path)
        # # print(self.a_star_game_states)
        # # Generate policy from A* path
        # self.v_solution, self.policy_solution = self.create_policy_valueTable_for_Astar(self.astar_path, self.a_star_game_states)
        # # print(self.v_solution)
        # # print(self.pai_solution)
        # # Expand reachable states from A* steps
        # reachable_states = self.collect_childs_states(self.a_star_game_states)
        # # print(reachable_states)
        # # Run value iteration on reachable space
        # self.V, self.policy = self.value_iteration(reachable_states)
        """Initialize controller for given game model."""
        self.original_game = game
        self.map, self.agent_pos, _, _, _ = self.original_game.get_current_state()

        # Compute A* path using helper method and correct directions
        self.astar_path = self.compute_Astar_path()
        self.solution_exist = len(self.astar_path) > 0

        # Create deterministic game state sequence along A* path
        self.a_star_game_states = self.create_game_object_by_action(self.astar_path)

        # Create policy and value tables based on A* path
        self.V_solution, self.pai_solution = self.create_policy_valueTable_for_Astar(self.astar_path, self.a_star_game_states)

        # Extract goal location from map
        pos = np.argwhere(self.map == pressure_plate.GOAL)
        self.goal = (pos[0][0], pos[0][1])

        # Expand reachable states from A* path
        reachable_states = self.collect_childs_states(self.a_star_game_states)

        # Run value iteration on reachable states
        self.V, self.policy = self.value_iteration(reachable_states)

   

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

###################################################################
        final_game = forced_games[-1]
        final_map = final_game.get_current_state()[0]
        plates_pressed = []
        for row in final_map:
            for cell in row:
                if 30 <= cell < 40:
                    plates_pressed.append(cell - 20)
        self.opened_doors = plates_pressed
################################################################

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
                #########################################################
                # יש הטוענים שעדיף עצירה ולא הפסקה כי זה קוטע מצבים

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


   
    # def value_iteration(self, reachable_states):
    #     V = {}
    #     pai = {}
    #     reachable = [game.get_current_state() for game in reachable_states]

    #     for state in reachable:
    #         V[tuple(state[0].flatten())] = 0
    #         pai[tuple(state[0].flatten())] = 'L'

    #     for _ in range(EPOCHS):
    #         new_V = {}
    #         for state in reachable:
    #             best_val = -np.inf
    #             best_act = 'U'
    #             for action in ACTIONS:
    #                 expected_val = 0.0
    #                 base = copy.deepcopy(self.original_game)
    #                 base._map = state[0].copy()
    #                 base._agent_pos = state[1]
    #                 base._steps = state[2]
    #                 base._done = state[3]
    #                 base._successful = state[4]

    #                 for i, act in enumerate(ACTIONS):
    #                     p = base._chosen_action_prob[action][i]
    #                     sim_game = copy.deepcopy(base)
    #                     sim_game._chosen_action_prob = DETERMINSITIC_PROBABLITIES
    #                     sim_game.submit_next_action(action)
    #                     next_state = sim_game.get_current_state()
    #                     h_next = tuple(next_state[0].flatten())
    #                     reward = sim_game.get_current_reward()
    #                     expected_val += p * (reward + DISCOUNT * V.get(h_next, 0))

    #                 if expected_val > best_val:
    #                     best_val = expected_val
    #                     best_act = action


    #             new_V[tuple(state[0].flatten())] = best_val
    #             pai[tuple(state[0].flatten())] = best_act

    #         V = new_V

    #     return V, pai

    def value_iteration(self, reachable_states):
        V = {}
        pai = {}
        reachable = [game.get_current_state() for game in reachable_states]

        for state in reachable:
            hashable_state = tuple(state[0].flatten())
            V[hashable_state] = self.init_V(state)
            pai[hashable_state] = 'U'

        for _ in range(EPOCHS):
            new_V = {}
            for state in reachable:
                hashable_state = tuple(state[0].flatten())
                best_val = -np.inf
                best_act = 'U'
                for action in ACTIONS:
                    expected_val = 0.0
                    base = copy.deepcopy(self.original_game)
                    base._map = state[0].copy()
                    base._agent_pos = state[1]
                    base._steps = state[2]
                    base._done = state[3]
                    base._successful = state[4]

                    for i, real_action in enumerate(ACTIONS):
                        p = base._chosen_action_prob[action][i]
                        sim_game = copy.deepcopy(base)
                        sim_game._chosen_action_prob = DETERMINSITIC_PROBABLITIES
                        sim_game.submit_next_action(action)
                        next_state = sim_game.get_current_state()
                        h_next = tuple(next_state[0].flatten())
                        reward = sim_game.get_current_reward()
                        reward += self.get_R(next_state)
                        expected_val += p * (reward + DISCOUNT * V.get(h_next, 0))

                    if expected_val > best_val:
                        best_val = expected_val
                        best_act = action

                new_V[hashable_state] = best_val
                pai[hashable_state] = best_act

            V = new_V

        return V, pai

    def get_R(self, state):
        R = 0
        hashable_state = tuple(state[0].flatten())
        if hashable_state in self.pai_solution:
            R += 100
        map = state[0]
        for door_number in self.opened_doors:
            positions = np.argwhere(map == door_number)
            for i, j in positions:
                walls = 0
                if i > 0 and map[i-1, j] == 99: walls += 1
                if i < map.shape[0]-1 and map[i+1, j] == 99: walls += 1
                if j > 0 and map[i, j-1] == 99: walls += 1
                if j < map.shape[1]-1 and map[i, j+1] == 99: walls += 1
                if walls >= 2:
                    R -= 50
        return R

    def init_V(self, state):
        agent_pos = state[1]
        return -abs(agent_pos[0] - self.goal[0]) - abs(agent_pos[1] - self.goal[1])

    

    # def choose_next_action(self, state):
    #     h_state = tuple(state[0].flatten())
    #     if h_state in self.policy_solution:
    #         # print(f"on optimal path - {self.pai_solution[h_state]}")
    #         return self.policy_solution[h_state]
    #     if h_state in self.policy:
    #         # print(f"on a child of optimal path - {self.pai[h_state]}")
    #         return self.policy[h_state]
    #     # print("random")
    #     return np.random.choice(["U","R","D","L"])
    def choose_next_action(self, state):
        h_state = tuple(state[0].flatten())

        # Case 1: On the optimal A* path
        if h_state in self.pai_solution:
            return self.pai_solution[h_state]

        # Case 2: In a reachable state from the A* path
        if h_state in self.policy:
            return self.policy[h_state]

        # Case 3: Not recognized — try to recover with new A* plan
        if not self.solution_exist:
            return np.random.choice(["U", "R", "D", "L"])

        current_map = state[0]
        problem = PressurePlateProblem(current_map)
        result = astar_search(problem)
        if result is None:
            self.solution_exist = False
            return np.random.choice(["U", "R", "D", "L"])

        # New A* path found — rebuild plan
        self.solution_exist = True
        result_node, _ = result
        path_nodes = list(reversed(result_node.path()))
        original_actions = [node.action for node in path_nodes if node.action is not None]
        corrected_actions = [self.correct_direction(a) for a in original_actions]
        sol_nodes = self.create_game_object_by_action(corrected_actions)
        self.V_solution, self.policy_solution = self.create_policy_valueTable_for_Astar(corrected_actions, sol_nodes)

        if h_state in self.policy_solution:
            return self.policy_solution[h_state]
        if h_state in self.policy:
            return self.policy[h_state]

        return np.random.choice(["U", "R", "D", "L"])
