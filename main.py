import gym
import QLearning_Agent
import Value_Iteration_Agent
from taxi import TaxiEnv
import numpy as np
import datetime
from tabulate import tabulate

# original 5x5 grid environment
env0 = gym.make('Taxi-v3')
# 5x5 grid - custom
env1 = TaxiEnv(1)
# 10x10 grid - custom
env2 = TaxiEnv(2)

# BLUE:current passenger location PURPLE:destination
# 0=south 1=north 2=east 3=west 4=pickup 5=dropoff

#let both agents play the game 100 times, return results
def run_experiment(policy, env, episodes, render):
    total_score, total_epochs, total_penalties, total_limit_reached = 0, 0, 0, 0
    #100 rounds of the game

    for _ in range(episodes):
        curr_state = env.reset()
        score, epochs, penalties, reward = 0, 0, 0, 0
        done = False

        while not (done or reward == 20):
            state, reward, done, info = env.step(np.argmax(policy[curr_state]))
            curr_state = state
            env.s = curr_state

            if reward == -10:
                penalties += 1

            score = score + reward
            epochs += 1
            if render == 1:
                env.render()

            if epochs == 100:
                total_limit_reached += 1
                break

        total_penalties += penalties
        total_epochs += epochs
        total_score += score

    average_score = total_score / episodes
    average_timesteps = total_epochs / episodes
    average_penalties = total_penalties / episodes
    return [average_score, average_timesteps, average_penalties, total_limit_reached]

# obtain and present results of experiment
def results(q_table, v_table, env, q_timing, v_timing):
    q_results = run_experiment(q_table, env, 100, 0)
    v_results = run_experiment(v_table[0], env,100, 0)
    results_table = [["average score", q_results[0], v_results[0]],
                     ["average no of time-steps", q_results[1], v_results[1]],
                     ["average no of penalties", q_results[2], v_results[2]],
                     ["no. of times max no of steps is reached", q_results[3], v_results[3]],
                     ["time take to train agent", q_timing, v_timing]]

    return results_table

# INITALISING AGENTS AND CONDUCTION EXPERIMENTS

# creating and training agents on original 5x5 grid and displaying results
print("ORIGINAL (5x5 grid) MAP RESULTS (over 10 episodes of the game):\n")
print("original map:")
env0.render()
print("training agents, might take a few minutes...")

value_agent = Value_Iteration_Agent.Value_Iteration_Agent(env0, theta=0.0001, discount_factor=0.99)
start_time = datetime.datetime.now()
value_policy = value_agent.update_values(env0)
end_time = datetime.datetime.now()
v_time = end_time - start_time

q_agent = QLearning_Agent.QLearning_Agent(env0, learning_rate=0.1, decay_rate=0.01, gamma=0.6)
start_time = datetime.datetime.now()
q_agent.train_agent(env0)
end_time = datetime.datetime.now()
q_time = end_time - start_time
Q_table = q_agent.Qtable
print("training done")
data = results(Q_table, value_policy, env0, q_time, v_time)
print(tabulate(data, headers=["original map (5x5 grid) results over 10 episodes", "Q-learning", "Value Iteration", ""]))

# using previously trained agents on custom, unseen 5x5 grid and displaying results
print("\nUNSEEN (5x5 grid) MAP RESULTS (over 10 episodes of the game):")
print("previously unseen map:")
env1.render()
data = results(Q_table, value_policy, env1, q_time, v_time)
print(tabulate(data, headers=["unseen map (5x5 grid) results over 10 episodes", "Q-learning", "Value Iteration", ""]))

# creating and training agents on custom 10x10 grid and displaying results
print("\n10x10 grid map:")
env2.render()
env2.render_to_file()
print("training agents, might take a few minutes...")
value_agent = Value_Iteration_Agent.Value_Iteration_Agent(env2, theta=0.0001, discount_factor=0.99)
start_time = datetime.datetime.now()
value_policy1 = value_agent.update_values(env2)
end_time = datetime.datetime.now()
v_time = end_time - start_time

q_agent = QLearning_Agent.QLearning_Agent(env2, learning_rate=0.1, decay_rate=0.01, gamma=0.6)
start_time = datetime.datetime.now()
q_agent.train_agent(env1)
end_time = datetime.datetime.now()
q_time = end_time - start_time

Q_table1 = q_agent.Qtable
print("training done\n")
data = results(Q_table1, value_policy1, env1, q_time, v_time)
print(tabulate(data, headers=["10x10 grid results over 10 episodes", "Q-learning", "Value Iteration", ""]))
print()
print()
print("Would you like to see the steps the two agents take in the original environment when playing the game?")
print("Type 'y' then press enter to see the steps, or type 'n' and press enter to exit the program.")
val = input("y or n: ")
print(val)
if val == 'y':
    print("------Q LEARNING AGENT-------")
    run_experiment(Q_table, env0, 1, 1)
    print()
    print("------VALUE ITERATION AGENT------")
    run_experiment(value_policy[0], env0, 1, 1)



