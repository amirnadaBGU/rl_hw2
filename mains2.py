import itertools
import random
import pyRDDLGym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from itertools import product
from matplotlib import pyplot as plt
import math


# Global variables
base_path ='./' # '../../Desktop/'
domain_file = base_path + 'jobs_domain.rddl'
instance_file = base_path + 'jobs_instance.rddl'

# GLOBALS

def generate_states():
    return list(product([True, False], repeat=5))

def generate_random_policy(states=generate_states()):
    policy = {}
    for state in states:
        state_tuple = tuple(state)  # Use tuple as dictionary key
        false_indices = [i for i, val in enumerate(state) if val == False]

        if not false_indices:
            policy[state_tuple] = [False] * len(state)  # No action possible
            continue

        action_index = random.choice(false_indices)  # Pick one False index to activate
        action = [False] * len(state)
        action[action_index] = True
        policy[state_tuple] = action
    return policy



def return_v_function(policy):
    def get_mu(action):
        mus = np.array([0.6, 0.5, 0.3, 0.7, 0.1])
        action = np.array(action, dtype=float)
        return np.dot(mus, action)

    def get_next_state(state, action):
        state = np.array(state, dtype=bool)
        action = np.array(action, dtype=bool)
        next_state = tuple(state | action)
        return next_state

    def get_total_cost(state):
        costs = [1,4,6,2,9]
        return sum(costs) - np.dot(costs, state)

    def update_value(states,index,action,values):
        state = states[index]
        mu = get_mu(action)
        next_state = get_next_state(state,action)
        next_state_index = states.index(next_state)
        new_value = get_total_cost(state) + values[index]*(1-mu) + values[next_state_index]*(mu)
        return new_value

    def init_values(states):
        values = []
        for s, state in enumerate(states):
            value = get_total_cost(state)
            values.append(value)
        return values
    epsilon = 0.05

    states = generate_states()
    values = init_values(states)
    temp_values = values.copy()
    value_history = [values.copy()]
    delta = float('inf')
    while delta > epsilon:
        values = temp_values.copy()
        for index,state in enumerate(states):
            temp_values[index] = update_value(states,index,policy[state],values)
        value_history.append(values.copy())
        delta = np.max(np.abs(np.array(values) - np.array(temp_values)))

    value_history = np.array(value_history)  # shape: (iterations, num_states)
    for i in range(len(states)):
        plt.plot(value_history[:, i], label=f"State {i}")
    plt.xlabel("Iteration")
    plt.ylabel("V(state)")
    plt.title("Value Function Convergence")
    plt.grid(True)
    plt.show()

    return values


if __name__ == '__main__':
    section_0 = False
    section_2 = True
    if section_0:
        # policy_evaluation()
        env = pyRDDLGym.make(domain=domain_file, instance=instance_file)
        # while True:
        #     avaliable_job_names = list(env.action_space.keys()) #give me all that is not done
        #     if avaliable job names i empty then break,
        #     else do some job according to your policy.

        job_names = list(env.action_space.keys()) #give me all that is not done
        tc = 0
        for job in job_names:
            action = {job: False for job in job_names}
            action[job] = True
            # Step the environment
            next_state, cost, _, _, _ = env.step(action)

            was_done = [next_state['was_done___j1'], next_state['was_done___j2']
                , next_state['was_done___j3'], next_state['was_done___j4']
                , next_state['was_done___j5']]

            print(was_done)
            print(cost)
    if section_2:
        random_policy = generate_random_policy()
        return_v_function(random_policy)



