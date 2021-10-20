import numpy as np
import random


class QLearning_Agent:
    def __init__(self, env, learning_rate, decay_rate, gamma):
        self.action_size = env.observation_space.n
        self.state_size = env.action_space.n
        self.Qtable = np.zeros([env.observation_space.n, env.action_space.n])

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay_rate = decay_rate

        self.episodes = 2000
        self.all_epochs = []
        self.all_penalties = []

    def train_agent(self, env):
        for i in range(1, self.episodes):
            state = env.reset()
            epochs = 0
            penalties = 0
            done = False

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    # explore action space
                    action = env.action_space.sample()
                else:
                    # exploit already existing values
                    action = np.argmax(self.Qtable[state])

                next_state, reward, done, info = env.step(action)
                old_value = self.Qtable[state, action]
                next_max = np.max(self.Qtable[next_state])

                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.gamma * next_max)
                self.Qtable[state, action] = new_value

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            # update epsilon so exploitation increases as time goes on
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon * np.exp(-self.decay_rate * epochs))

        print("training done")
