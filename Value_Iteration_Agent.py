import numpy as np


class Value_Iteration_Agent:
    def __init__(self, env, theta, discount_factor):
        self.V = np.zeros(env.nS)
        env.nS = env.observation_space.n
        env.nA = env.action_space.n
        self.theta = theta
        self.discount_factor = discount_factor
        self.policy = np.zeros([env.nS, env.nA])

    def one_step_lookahead(self, state, env):
        # helper function to calculate the value for all action in given state
        A = np.zeros(env.nA)
        for act in range(env.nA):
            for prob, next_state, reward, done in env.P[state][act]:
                A[act] += prob * (reward + self.discount_factor * self.V[next_state])
        return A

    def update_values(self, env):
        while True:
            # checks for convergence of values
            delta = 0
            for state in range(env.nS):
                # looks at possible next steps
                act_values = self.one_step_lookahead(state, env)
                # get best action value of the state
                best_act_value = np.max(act_values)
                delta = max(delta, np.abs(best_act_value - self.V[state]))  # find max delta across all states
                self.V[state] = best_act_value  # update value to best action value
            # if max improvement less than threshold
            if delta < self.theta:
                break

        # final value iteration policy table
        for state in range(env.nS):
            act_val = self.one_step_lookahead(state, env)
            best_action = np.argmax(act_val)
            self.policy[state][best_action] = 1

        return self.policy, self.V


