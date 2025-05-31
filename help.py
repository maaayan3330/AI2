import zuma
import copy
import re
import random

id = ["315249003"]


class Controller:
    """This class is a controller for a Zuma game."""

    def __init__(self, game: zuma.Game):
        """Initialize controller for given game model.
        This method MUST terminate within the specified timeout.
        """
        self.game = game
        self.copy_game = copy.deepcopy(game)
        self.model = game.get_model()
        self._chosen_action_prob = self.model['chosen_action_prob']
        self._color_pop_reward = self.model['color_pop_reward']
        self._color_pop_prob = self.model['color_pop_prob']
        self._color_not_finished_punishment = self.model['color_not_finished_punishment']
        self._finished_reward= self.model['finished_reward']

    def choose_next_action(self):
        """Choose next action for Zuma given the current state of the game."""
        current_line, current_ball, steps, max_steps = self.game.get_current_state()
        remain_steps = max_steps - steps - 1
        potential_actions = self.relvant_index(current_line, current_ball, remain_steps, max_steps)

        if len(potential_actions) == 1:
            return potential_actions[0]

        best_action = -1
        best_reward = float('-inf')

        for action in potential_actions:
            total_reward = 0

            for position in range(-1, len(current_line) + 1):
                if position == -1:
                    immediate_reward = self._finished_game(current_line, remain_steps)
                else:
                    simulated_line = current_line[:]
                    simulated_line.insert(action, current_ball)
                    immediate_reward = self._remove_group(simulated_line, action, remain_steps)

                probability = (
                    self._chosen_action_prob[current_ball] 
                    if action == position 
                    else (1 - self._chosen_action_prob[current_ball]) / (len(current_line) + 1)
                )
                total_reward += probability * immediate_reward

            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action

        return best_action

    def relvant_index(self, line, ball, remain_steps, max_steps):
        index = []
        i = 0
        while i < len(line):
            if ball == line[i]:
                index.append(i)
                while i < len(line) and ball == line[i]:
                    i += 1
            else:
                i += 1
        if not index:
            if remain_steps <= max_steps/8:
                index.append(-1)
            else:
                index.append(random.choice([-1, 0]))
        return index
    

    def _remove_group(self, line, idx, steps, reward=0):

        burstable = re.finditer(r'1{3,}|2{3,}|3{3,}|4{3,}', ''.join([str(i) for i in line]))
        new_line = line.copy()
         
        for group in burstable:
            if idx in range(group.span()[0], group.span()[1]):
                finshed_reward = self._finished_game(new_line, steps)
                no_pop_reward = reward + (1-self._color_pop_prob[line[group.start()]]) * (finshed_reward)
                pop_reward = (self._color_pop_reward['3_pop'][line[group.start()]] +
                                (group.span()[1] - group.span()[0] - 3) *
                                self._color_pop_reward['extra_pop'][line[group.start()]])
                new_line = line[:group.span()[0]] + line[group.span()[1]:]
                idx = group.span()[0]

                return no_pop_reward + self._color_pop_prob[line[group.start()]] * (pop_reward + self._remove_group(new_line, idx, steps, pop_reward))


        return self._finished_game(new_line, steps)



    def _finished_game(self, line, steps):
        """
        Rewards or punishes for any leftovers in the line
        """
        if not line:
            return self._finished_reward if steps == 0 else 0        

        if steps == 0:  
            penalty = sum(line.count(ball) * penalty for ball, penalty in self._color_not_finished_punishment.items())
            return -penalty
        return 0
                