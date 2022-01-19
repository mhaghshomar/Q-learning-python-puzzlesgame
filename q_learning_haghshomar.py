from utils import *
import numpy as np
import random

# M is Moon, S is Sun, B is Blank
states = list(4*'M'+4*'S'+'B')
states = sorted(unique_perms(states))
states = list(np.array(list(chunks(state, 3)))for state in states)
moves = ['up', 'right', 'down', 'left']
init_state = np.array([['S', 'S', 'S'],
                       ['M', 'S', 'M'],
                       ['M', 'B', 'M']])
final_state = np.array([['B', 'M', 'S'],
                        ['M', 'S', 'M'],
                        ['S', 'M', 'S']])


class Board:
    def __init__(self, first_state=init_state):
        self.states = states
        self.state = first_state
        self.final_state = final_state

    def next_state(self, action):
        recent_blank = np.where(self.state == 'B')
        i, j = recent_blank[0][0], recent_blank[1][0]

        if (i == 0 and action == 'up' or
                i == 2 and action == 'down' or
                j == 0 and action == 'left' or
                j == 2 and action == 'right'):
            return None

        if action == 'up':
            self.state[i][j], self.state[i -
                                         1][j] = self.state[i - 1][j], self.state[i][j]
        if action == 'right':
            self.state[i][j], self.state[i][j +
                                            1] = self.state[i][j + 1], self.state[i][j]
        if action == 'down':
            self.state[i][j], self.state[i +
                                         1][j] = self.state[i + 1][j], self.state[i][j]
        if action == 'left':
            self.state[i][j], self.state[i][j -
                                            1] = self.state[i][j - 1], self.state[i][j]

        return self.state

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_all_states(self):
        return self.states

    def get_final_state(self):
        return self.final_state


class Game:
    def __init__(self, initial_state, alpha, gamma):
        self.game = Board(initial_state)
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = self.set_Qtable()
        self.rewards = self.set_reward()
        self.path = [initial_state]
        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.01

    def set_reward(self):
        rewards = dict()
        states = self.game.get_all_states()
        goal_state = self.game.get_final_state()
        for state in states:
            s = 0
            s += np.sum(state == goal_state)
            rewards[tuple(state.flatten())] = s
            if np.array_equal(state, goal_state):
                rewards[tuple(state.flatten())] = 100
        return rewards

    def set_Qtable(self):
        states = self.game.get_all_states()
        q_table = dict()
        for state in states:
            q_table[tuple(state.flatten())] = {
                'up': 0, 'right': 0, 'down': 0, 'left': 0}
        return q_table

    def best_action(self, state):
        maximum = 0
        move = None
        for i in self.q_table[tuple(state.flatten())]:
            if self.q_table[tuple(state.flatten())][i] > maximum:
                maximum = self.q_table[tuple(state.flatten())][i]
                move = i
        return [move, maximum]

    def choose_action(self, cur_state):
        self.exploration_rate_threshold = random.uniform(0, 1)
        if (self.exploration_rate_threshold > self.exploration_rate and np.argmax(self.q_table[tuple(cur_state.flatten())]) !=0 ) :
            action, score = self.best_action(cur_state)
            print(action)

        else:
            action = random.choice(moves)
        return action

    def run(self):
        number_of_moves = 0
        while True:
            cur_state = self.game.get_state()
            action = self.choose_action(cur_state)
            next_state = None
            while next_state is None:
                action = self.choose_action(cur_state)
                next_state = self.game.next_state(action)
            number_of_moves += 1
            self.path.append(action)
            self.q_table[tuple(cur_state.flatten())][action] = \
                (1 - self.alpha) * self.q_table[tuple(cur_state.flatten())][action] + \
                self.alpha * \
                (self.rewards[tuple(next_state.flatten())] + self.gamma *
                 self.best_action(next_state)[1])

            if np.array_equal(next_state, self.game.get_final_state()):
                print('win in {} moves'.format(number_of_moves))
                for i in self.path:
                    print(i)
                if number_of_moves < 50:
                    return True
            if number_of_moves > 50:
                return False

            self.exploration_rate = self.min_exploration_rate + (
                        self.max_exploration_rate - self.min_exploration_rate) * np.exp(
                -self.exploration_decay_rate * number_of_moves)

        return self.path

    def new_path(self):
        self.path = list(init_state.flatten())

    def show(self):
        for i in self.q_table:
            print("state: {}".format(i))


Agent = Game(initial_state=init_state, alpha=0.1, gamma=0.9)

for i in range(1000):
    Agent.run()
    Agent.game.set_state(init_state)
Agent.show()
print(' ' * 120)
Agent.game.set_state(init_state)
Agent.new_path()
Agent.run()
