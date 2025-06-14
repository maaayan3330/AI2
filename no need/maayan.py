import pressure_plate
from ex1 import PressurePlateProblem
from search import astar_search
import numpy as np

id = ["212437453"]

AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3
WALL = 99
FLOOR = 98

# rewards
A_STAR_REWARD = 4
DEFAULT_REWARD = 0
WALL_REWARD = -999

#
DISCOUNT = 0.9
MAX_DEPTH = 5

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
        self.astar_path = self.compute_astar_path()
        print("🧭 Forced A* path:", self.astar_path)
        self.step_counter = 0
        self.rows , self.cols = self.map.shape

        # get the steps that match the actions
        self.path_positions = self.discover_path_steps()

        # inital the map with V(0) - by this rules:
        # 1 - wall will be infinite  | 2- step from path - positive num | door for now will be 0
        self.reward_map = self.initialize_reward_map()
        # print("🎯 Reward Map (V0):")
        # for row in self.reward_map:
        #     print(" ".join(f"{val:4}" for val in row))


      
    # this func - convert the steps of the A* path in to cells in the map
    def discover_path_steps(self):
        path_positions = []
        row, col = self.agent_pos
        # print("📍 A* path steps:")
        for i, action in enumerate(self.astar_path):
            dr, dc = DIRECTION[action]
            row += dr
            col += dc
            path_positions.append((row, col))
            # print(f"Step {i}: action {action} -> position ({row}, {col})")
        return path_positions
    
    # this func - inital the first V(0) rewards 
    def initialize_reward_map(self):
        # make a all map empty
        reward_map = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        # for faster search
        path_set = set(self.path_positions)

        for i in range(self.rows):
            for j in range(self.cols):
                if self.map[i][j] == WALL:
                     # if it is a wall so i dont eant the agaent to go there so - reward infinte
                    reward_map[i][j] = WALL_REWARD
                elif (i, j) in path_set:
                    # if this cell is part of the A* path-big reward
                    reward_map[i][j] = A_STAR_REWARD
                else:
                    reward_map[i][j] = DEFAULT_REWARD
        return reward_map

    # A*
    def compute_astar_path(self):
        current_map, _, _, _, _ = self.original_game.get_current_state()
        problem = PressurePlateProblem(current_map)
        result_node, _ = astar_search(problem)
        if result_node is None:
            # print("\u274C No A* path found.")
            return []
        path_nodes = list(reversed(result_node.path()))
        return [node.action for node in path_nodes if node.action is not None]

    
    def value(self, pos, depth):
       pass


    
    # do the policy & return the best action
    def choose_next_action(self, state):
        _, _, _, done, _ = state
        if done:
            # print("✅ Game is done. No further actions.")
            raise SystemExit()  # או return 'U' אם חייבים להחזיר פעולה

        if self.step_counter < len(self.astar_path):
            action = self.astar_path[self.step_counter]
            # print(f"Step {self.step_counter}: forced action {action}")
            self.step_counter += 1
            return action
        else:
            # print(f"⚠️ Reached end of path at step {self.step_counter}.")
            return 'U'  # פעולה נייטרלית שלא מקדמת

