import numpy as np

class QLearningAgent:
    def __init__(self, lr=0.1, gamma=0.95, epsilon=0.1):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = 0.99  # Decay rate for epsilon
        self.q_table = {}

    def get_state(self, val_loss, current_lr):
        return (round(val_loss.item(), 2), round(current_lr, 5))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            self.epsilon *= self.epsilon_decay  # Decay epsilon
            return np.random.choice([-1, 0, 1])
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        return np.argmax(self.q_table[state]) - 1

    def update_q_values(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action + 1]
        self.q_table[state][action + 1] += self.lr * td_error
