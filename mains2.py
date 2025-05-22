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
COSTS = [1, 4, 6, 2, 9]
MUS = np.array([0.6, 0.5, 0.3, 0.7, 0.1])
EPSILON = 0.005

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

def generate_cost_policy(states=generate_states()):
    policy = {}
    for state in states:
        state_tuple = tuple(state)  # Use tuple as dictionary key
        false_indices = [i for i, val in enumerate(state) if val == False]

        if not false_indices:
            policy[state_tuple] = [False] * len(state)  # No action possible
            continue

        max_cost = 0
        action_index = 0

        for i in false_indices:
            if COSTS[i] > max_cost:
                action_index = i
                max_cost = COSTS[i]

        action = [False] * len(state)
        action[action_index] = True
        policy[state_tuple] = action
    return policy

def get_mu(action):
    #mus = np.array([1, 1, 1, 1, 1])
    action = np.array(action, dtype=float)
    return np.dot(MUS, action)

def get_next_state(state, action):
    state = np.array(state, dtype=bool)
    action = np.array(action, dtype=bool)
    next_state = tuple(state | action)
    return next_state

def get_total_cost(state):
    return sum(COSTS) - np.dot(COSTS, state)

def calc_value (states,index,action,values):
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

def evaluate_policy(policy):
    states = generate_states()
    values = init_values(states)
    new_values = values.copy()
    value_history = [values.copy()]
    delta = float('inf')

    while delta > EPSILON:
        for index, state in enumerate(states):
            new_values[index] = calc_value(states, index, policy[state], values)
        delta = np.max(np.abs(np.array(values) - np.array(new_values)))
        value_history.append(new_values.copy())
        values = new_values.copy()

    value_history = np.array(value_history)  # shape: (iterations, num_states)
    for i in range(len(states)):
        plt.plot(value_history[:, i], label=f"State {i}")
    plt.xlabel("Iteration")
    plt.ylabel("V(state)")
    plt.title("Value Function Convergence")
    plt.grid(True)
    plt.show()

    return values

def policy_iteration(initial_policy):

    def find_best_action(states, index, values):
        state = states[index]
        false_indices = [i for i, val in enumerate(state) if not val]
        min_Q = float('inf')
        best_action = [False] * len(state)

        if not false_indices:
            best_action = [False] * len(state)
            return best_action

        for i in false_indices:
            action = [False] * len(state)
            action[i] = True

            q = calc_value(states, index, action, values)

            if q < min_Q:
                min_Q = q
                best_action = action

        return best_action

    delta = float('inf')
    states = generate_states()

    # LOOP
    values = init_values(states)
    new_values = values.copy()
    policy = initial_policy.copy()
    new_policy = policy.copy()
    value_history = [values.copy()]

    while delta > EPSILON:

        # Update values according to policy
        for index, state in enumerate(states):
            new_values[index] = calc_value(states, index, policy[state], values)

        # Policy improvment
        for index, state in enumerate(states):
            best_action = find_best_action(states, index, new_values)
            new_policy[state] = best_action

        delta = np.max(np.abs(np.array(values) - np.array(new_values)))
        value_history.append(new_values.copy())

        values = new_values.copy()
        policy = new_policy.copy()


    # Plot value function convergence
    value_history = np.array(value_history)
    plt.figure()
    # for i in range(value_history.shape[1]):
    #     plt.plot(value_history[:, i], label=f"State {i}")

    plt.plot(value_history[:, -1], label=f"State 0")

    plt.xlabel("Iteration")
    plt.ylabel("V(state)")
    plt.title("Value Function Convergence During Policy Iteration")
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return policy, values

if __name__ == '__main__':
    section_0 = False
    section_2 = True
    section_3 = True
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
        states = generate_states()
        for k,state in enumerate(states):
                print(k)
                print(state)
                print('-----')
        random.seed(42)
        # random_policy = generate_random_policy()
        cost_policy = generate_cost_policy()
        evaluate_policy(cost_policy)
    if section_3:
        cost_policy = generate_cost_policy()
        policy_iteration(initial_policy=cost_policy)



