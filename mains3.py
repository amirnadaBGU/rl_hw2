import itertools
import random
import re

# from pipes import stepkinds

import pyRDDLGym
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from itertools import product
from matplotlib import pyplot as plt
import math


import mains2

# Globals
EPSILON = 0.00000005


# Global variables
base_path ='./' # '../../Desktop/'
domain_file = base_path + 'jobs_domain.rddl'
instance_file = base_path + 'jobs_instance.rddl'

def generate_states(num_jobs):
    return list(product([True, False], repeat=num_jobs))

def generate_cost_policy(states):
    costs = [1, 4, 6, 2, 9]
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
            if costs[i] > max_cost:
                action_index = i
                max_cost = costs[i]

        action = [False] * len(state)
        action[action_index] = True
        policy[state_tuple] = action
    return policy

def convert_action_to_sim_action_dict(job_names,bool_action):
    action_dict = {job: val for job, val in zip(job_names, bool_action)}
    return action_dict

def convert_sim_state_to_bool_state(sim_state):
    was_done = {}
    for key in sim_state.keys():
        match = re.match(r'was_done___j(\d+)', key)
        if match:
            index = int(match.group(1)) - 1  # convert j1-based to 0-based
            was_done[index] = sim_state[key]
    return tuple([was_done[i] for i in sorted(was_done)])

def policy_evaluation(sub_section):
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    delta = float('inf')
    job_names = list(env.action_space.keys())  # give me all that is not done
    states = generate_states(len(job_names))
    values = {state: 0 for state in states}
    visits = {state: 0 for state in states}
    policy = generate_cost_policy(states)

    max_norm = []

    v_pi_c_raw = mains2.evaluate_policy(policy, False)
    v_pi_c = dict(zip(states, v_pi_c_raw))

    v_pi_c_so = v_pi_c[tuple([False]*len(states[0]))]
    v_TD_so = []


    counter = 0
    while delta > EPSILON:
        counter+=1
        # Initial state
        state = env.reset()[0]
        state = convert_sim_state_to_bool_state(state)
        old_values = values.copy()

        for i in range(env.horizon):
            action = policy[state]
            action = convert_action_to_sim_action_dict(job_names, action)
            # Step the environment
            next_state, cost, done, _, _ = env.step(action)
            # print(next_state)
            next_state = convert_sim_state_to_bool_state(next_state)

            visits[state] += 1

            alpha = 0.01

            if sub_section == "sub_section_1":
                alpha = 1/visits[state]

            elif sub_section == "sub_section_2":
                alpha = 0.01

            elif sub_section == "sub_section_3":
                alpha = 10/(100 + visits[state])

            current_value = values[state]
            values[state] = current_value + alpha * (cost + values[next_state] - current_value)

            state = tuple(list(next_state).copy())

            if done:
                break

        if counter>30000:
            break

        valid_states = [s for s in values if visits[s] > 0 and s in v_pi_c]
        if valid_states:
            max_norm.append(max(abs(v_pi_c[s] - values[s]) for s in valid_states))
        else:
            max_norm.append(0)  # Fallback in case nothing was visited yet
            print('i am here')

        v_TD_so.append(values[tuple([False] * len(states[0]))])
        delta = max(abs(old_values[s] - values[s]) for s in states)
        if counter % 1000 == 0:
            print(counter)
            print(delta)
        if delta <= EPSILON:
            print(f"Stop: {delta}")
        # print(delta)
        # print("visits: ")



    # Plot V(π_TD) and V(π_c) State Values
    plt.figure()
    plt.plot([v_pi_c[state] for state in states], label="V(π_c)", marker='o', linewidth=1)
    plt.plot([values[state] for state in states], label="V(π_TD)", marker='x', linewidth=1)
    plt.xlabel("State Index")
    plt.ylabel("Value")
    plt.title("V(π_TD) and V(π_c) State Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    delta_s0 = abs(v_TD_so - v_pi_c_so)
    # Plot V(π_TD)(s0) vs V(π_c)(s0)
    plt.figure()
    plt.plot(delta_s0, marker='o', linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Absolute Error")
    plt.title("|V(π_TD)(S0) - V(π_c)(S0)| VS Episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if sub_section == "sub_section_1":
        alpha = "1/n_visits"
    elif sub_section == "sub_section_2":
        alpha = "0.01"
    elif sub_section == "sub_section_3":
        alpha = "10/(100+n_visits)"
    plt.plot(max_norm, label=f"{sub_section}, alpha: {alpha}")
    plt.title(r"$\|V^{\pi_c} - \hat{V}_{TD}\|_\infty$ over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Infinity Norm Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return max_norm

random.seed(0)
np.random.seed(0)


# Run evaluations and collect max-norm error lists
max_norm_1 = policy_evaluation("sub_section_1")
# max_norm_2 = policy_evaluation("sub_section_2")
# max_norm_3 = policy_evaluation("sub_section_3")








