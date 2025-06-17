# I used AI tools including ChatGPT, Gemini, and GitHub Copilot during the development of this code.
# I also discussed ideas and received input from peers in the course WhatsApp group.
# I also collaborated with Maayan Ifargan in exchanging ideas and discussing approaches.
import pressure_plate
import numpy as np
from collections import deque
import copy
from ex1 import PressurePlateProblem
from search import astar_search

id = ["211548045"]


""" Rules """
BLANK = 0
WALL = 99
FLOOR = 98
AGENT = 1
GOAL = 2
AGENT_ON_GOAL = 3


forced_actions = {'U': [1, 0, 0, 0],
                  'L': [0, 1, 0, 0],
                  'R': [0, 0, 1, 0],
                  'D': [0, 0, 0, 1]}


class Controller:
    """This class is a controller for a pressure plate game."""

    def __init__(self, game: pressure_plate.Game):
        """Initialize controller for given game model.
        """
        self.original_game = game
        self.map = game.get_current_state()[0]
        p = PressurePlateProblem(self.map)
        result = astar_search(p)
        self.solution_exist = False
        if result is None:
            solution = []
        else:
            self.solution_exist = True
            Node, expanded = result
            solve = Node.path()[::-1]
            solution = [pi.action for pi in solve][1:]
        swap = {'U': 'D', 'D': 'U'}
        new_solution = [swap.get(a, a) for a in solution]
        sol_nodes = self.create_sol_nodes(new_solution)
        self.V_solution, self.pai_solution = self.create_sol_policy(new_solution, sol_nodes)
        pos = np.argwhere(self.map == pressure_plate.GOAL)
        self.goal = (pos[0][0], pos[0][1])
        reachable_states = self.create_reachable_nodes(sol_nodes)
        # print(len(reachable_states))
        self.V, self.pai = self.value_iteration(reachable_states=reachable_states)

    def create_sol_policy(self, new_solution, sol_nodes):
        V = {}
        pai = {}
        for i, game in enumerate(sol_nodes[:-1]):
            current_state = game.get_current_state()
            hashable_state = tuple(current_state[0].flatten())
            V[hashable_state] = i
            pai[hashable_state] = new_solution[i]
        final_game = sol_nodes[-1]
        final_state = final_game.get_current_state()
        hashable_state = tuple(final_state[0].flatten())
        V[hashable_state] = len(sol_nodes)
        pai[hashable_state] = "U"
        return V, pai
            
    def create_reachable_nodes(self, sol_nodes, h = 4):
        reachable_nodes = []
        for game in sol_nodes:
            visited = set()
            q = deque()
            q.append((copy.deepcopy(game),0))
            while q:
                current_game, depth = q.popleft()
                if depth >= h:
                    break
                current_state = current_game.get_current_state()
                map = current_state[0]
                hashable_map = tuple(map.flatten())
                if hashable_map in visited:
                    continue
                visited.add(hashable_map)
                reachable_nodes.append(current_game)
                for action in ["U", "L", "R", "D"]:
                    next_state = copy.deepcopy(current_game)
                    next_state._chosen_action_prob = forced_actions
                    next_state.submit_next_action(action)
                    q.append((next_state, depth + 1))
        return reachable_nodes
            
    def create_sol_nodes(self, new_solution):
        copied_game = copy.deepcopy(self.original_game)
        sol_nodes = []
        sol_nodes.append(copy.deepcopy(self.original_game))
        copied_game._chosen_action_prob = forced_actions
        for action in new_solution:
            copied_game.submit_next_action(action)
            sol_nodes.append(copy.deepcopy(copied_game))
            
        final_game = sol_nodes[-1]
        final_map = final_game.get_current_state()[0]
        plates_pressed = []
        for row in final_map:
            for cell in row:
                if cell >= 30 and cell < 40:  
                    plates_pressed.append(cell - 20)

        self.opened_doors = plates_pressed
        return sol_nodes
        
    def choose_next_action(self, state):
        """Choose next action for a pressure plate game given the current state of the game.
        """
        hashable_state = tuple(state[0].flatten())
        # reachable_states = self.limited_bfs(state)
        # self.V, self.pai = self.value_iteration(reachable_states=reachable_states)

        if hashable_state in self.pai_solution:
            temp = self.pai_solution[hashable_state]
            # print(temp)
            return self.pai_solution[hashable_state]
        if hashable_state in self.pai:
            temp = self.pai[hashable_state]
            # print(temp)
            return self.pai[hashable_state]
        if not self.solution_exist:
            return np.random.choice(["U", "R", "D", "L"])
        p = PressurePlateProblem(state[0])
        result = astar_search(p)
        if result is None:
            self.solution_exist = False
            return np.random.choice(["U", "R", "D", "L"])
        else:
            self.solution_exist = True
            Node, expanded = result
            solve = Node.path()[::-1]
            solution = [pi.action for pi in solve][1:]
            swap = {'U': 'D', 'D': 'U'}
            new_solution = [swap.get(a, a) for a in solution]
            sol_nodes = self.create_sol_nodes(new_solution)
            self.V_solution, self.pai_solution = self.create_sol_policy(new_solution, sol_nodes)
            # reachable_states = self.create_reachable_nodes(sol_nodes)
            # self.V, self.pai = self.value_iteration(reachable_states=reachable_states)
            if hashable_state in self.pai_solution:
                return self.pai_solution[hashable_state]
            if hashable_state in self.pai:
                return self.pai[hashable_state]
            return np.random.choice(["U", "R", "D", "L"])


        # if hashable_state in self.pai:
        #     return self.pai[hashable_state]
        # return np.random.choice(["U", "R", "D", "L"])

        # return np.random.choice(["U", "R", "D", "L"])

    def value_iteration(self, gamma=0.9, epochs=15, reachable_states=None):
        reachable = [game.get_current_state() for game in reachable_states]
        V = {}
        pai = {}
        for state in reachable:
            hashable_state = tuple(state[0].flatten())
            V[hashable_state] = self.init_V(state)
            pai[hashable_state] = "U"
        for i in range(epochs):
            new_V = {}
            for state in reachable:
                hashable_state = tuple(state[0].flatten())
                best_value = -np.inf
                best_action = "U"
                for action in ["U", "L", "R", "D"]:
                    E_V = 0.0
                    map, agent_pos, steps, done, successful = state
                    copied_game = copy.deepcopy(self.original_game)
                    copied_game._agent_pos = agent_pos
                    copied_game._steps = steps
                    copied_game._done = done
                    copied_game._successful = successful
                    copied_game._map = map.copy()
                    for i, real_action in enumerate(["U", "L", "R", "D"]):
                        p = copied_game._chosen_action_prob[action][i]
                        copied_game2 = copy.deepcopy(copied_game)
                        copied_game2._chosen_action_prob = forced_actions
                        copied_game2.submit_next_action(action)
                        next_state = copied_game2.get_current_state()
                        hashable_next_state = tuple(next_state[0].flatten())
                        V_next = V.get(hashable_next_state, 0)
                        R = copied_game2.get_current_reward()
                        R += self.get_R(next_state)
                        E_V += p * (R + gamma * V_next)
                    if E_V > best_value:
                        best_value = E_V
                        best_action = action
                new_V[hashable_state] = best_value
                pai[hashable_state] = best_action
            V = new_V
        return V, pai

    def get_R(self, state):
        R = 0
        hashable_state = tuple(state[0].flatten())
        if hashable_state in self.pai_solution:
            R = R + 100
        map = state[0]
        for door_number in self.opened_doors:
            positions = np.argwhere(map == door_number)
            for pos in positions:
                i, j = pos
                walls = 0
                if i > 0 and map[i-1, j] == 99:
                    walls += 1
                if i < map.shape[0]-1 and map[i+1, j] == 99:
                    walls += 1
                if j > 0 and map[i, j-1] == 99:
                    walls += 1
                if j < map.shape[1]-1 and map[i, j+1] == 99:
                    walls += 1
                if walls >= 2:
                    R = R - 50
        return R

    def init_V(self, state):
        agent_pos = state[1]
        distance = abs(agent_pos[0] - self.goal[0]) + abs(agent_pos[1] - self.goal[1])
        return -distance

    