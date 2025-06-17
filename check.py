import pressure_plate
import ex2


def solve(game: pressure_plate.Game):
    policy = ex2.Controller(game)
    for i in range(game.get_max_steps()):
        game.submit_next_action(chosen_action=policy.choose_next_action(game.get_current_state()))
        if game.get_current_state()[3]:
            break
    print('Game result:\n\tMap state ->\n', game.get_current_state()[0], '\n\tFinished in', game.get_current_state()[2],
          'Steps.\n\tReward result->',game.get_current_reward())
    print("Game finished ", "" if game.get_current_state()[-1] else "un", "successfully.", sep='')
    game.show_history()
    return game.get_current_reward()


# ["U", "L", "R", "D"]
example = {'chosen_action_prob': {'U': [0.9, 0.05, 0.05, 0], 'L': [0.1, 0.8, 0.075, 0.025],
                                  'R': [0.05, 0.05, 0.85, 0.05], 'D': [0.05, 0.1, 0.15, 0.7]},
           'finished_reward': 350,
           'opening_door_reward': {10: 5, 11: 7, 12: 9, 13: 11, 14: 13, 15: 15, 16: 17, 17: 19, 18: 21, 19: 23},
           'step_punishment': -2,
           'seed': 42}

example2 = {'chosen_action_prob': {'U': [0.6, 0.05, 0.05, 0.3], 'L': [0, 0.9, 0.075, 0.025],
                                   'R': [0.25, 0.2, 0.3, 0.25], 'D': [0.05, 0.128, 0.15, 0.672]},
            'finished_reward': 200,
            'opening_door_reward': {10: -3, 11: 2, 12: 15, 13: -6, 14: 3, 15: -10, 16: 17, 17: 0, 18: 1, 19: -2},
            'step_punishment': -2,
            'seed': 42}
# experiment with the seed as well
problem1 = (
    (99,99,99,99,99,99),
    (99,2 ,40,98,98,99),
    (99,99,99,10,98,99),
    (99,98,98,98,98,99),
    (99,20,98,98,1 ,99),
    (99,99,99,99,99,99),
)

problem2 = (
    (99,99,99,99,99,99,99,99,99,99,99,99,99,99,99),
    (99,98,98,98,99,99,99,99,99,99,99,99,99,99,99),
    (99,98,99,98,99,99,99,99,99,99,98,98,99,99,99),
    (99,98,99,98,98,99,25,98,99,99,98,98,98,99,99),
    (99,98,99,98,2 ,45,98,98,98,98,98,98,98,99,99),
    (99,98,99,99,99,99,98,98,99,99,99,42,99,99,99),
    (99,98,98,98,98,99,99,99,99,99,22,98,98,99,99),
    (99,99,99,99,98,99,98,98,98,99,98,98,98,99,99),
    (99,98,98,98,98,99,12,98,98,99,98,98,98,99,99),
    (99,98,99,99,23,98,98,15,98,99,99,41,99,99,99),
    (99,98,99,99,98,98,98,98,98,99,20,98,98,98,99),
    (99,98,99,99,98,98,99,98,98,99,98,98,10,98,99),
    (99,98,99,99,98,13,98,98,98,40,11,98,98,98,99),
    (99,98,43,98,98,98,98,98,98,99,21,98,98,1 ,99),
    (99,99,99,99,99,99,99,99,99,99,99,99,99,99,99),)

problem5 = (
    (99, 99, 99, 99, 99, 99, 99, 99, 99, 99),
    (99, 22, 98, 98, 98, 12, 98, 98, 1, 99),
    (99, 98, 98, 11, 98, 10, 98, 99, 98, 99),
    (99, 98, 98, 99, 42, 99, 99, 99, 98, 99),
    (99, 98, 99, 99, 98, 98, 98, 99, 40, 99),
    (99, 41, 99, 98, 98, 98, 21, 99, 98, 99),
    (99, 98, 99, 98, 98, 98, 20, 99, 98, 99),
    (99, 98, 99, 99, 99, 99, 99, 99, 98, 99),
    (99, 2, 98, 98, 98, 98, 98, 98, 98, 99),
    (99, 99, 99, 99, 99, 99, 99, 99, 99, 99),
)

example3 = {
    'chosen_action_prob': {
        'U': [0.5, 0.15, 0.2, 0.15],
        'L': [0.2, 0.5, 0.15, 0.15],
        'R': [0.15, 0.2, 0.5, 0.15],
        'D': [0.15, 0.15, 0.2, 0.5]
    },
    'finished_reward': 180,
    'opening_door_reward': {10: 25, 11: 20, 12: 15, 13: 10, 14: 30, 15: 20, 16: 10, 17: 15, 18: 20, 19: 25},
    'step_punishment': -0.5,
    'seed': 222
}



def main():
    debug_mode = True
    game = pressure_plate.create_pressure_plate_game((100, problem1, example2, debug_mode))
    solve(game)
    game2 = pressure_plate.create_pressure_plate_game((200, problem2, example, debug_mode))
    solve(game2)
    # game3 = pressure_plate.create_pressure_plate_game((200, problem5, example, debug_mode))
    # solve(game3)


if __name__ == "__main__":
    main()
