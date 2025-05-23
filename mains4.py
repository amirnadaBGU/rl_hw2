import copy
import itertools
import random
import re

# from pipes import stepkinds

import pyRDDLGym
import numpy as np
import matplotlib
from plotly.figure_factory.utils import list_of_options

matplotlib.use('TkAgg')
from itertools import product
from matplotlib import pyplot as plt
import math


import mains2

# Globals
EPSILON = 0.00000005
EXPLORATION = 0.1

# Global variables
base_path ='./' # '../../Desktop/'
domain_file = base_path + 'jobs_domain.rddl'
instance_file = base_path + 'jobs_instance.rddl'

def generate_c_mu_policy():
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    job_names = list(env.action_space.keys())  # give me all that is not done
    states = generate_states(len(job_names))
    policy ={}
    for state in states:
        policy[state] = mains2.c_mu_action(state)
    return policy

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

def generate_actions_per_state(state):
    list_of_actions = []
    false_indices = [i for i, val in enumerate(state) if val == False]
    if not false_indices: # All jobs are True, then the action is all False
        action = tuple(([False] * len(state)))  # No action possible
        return [action]

    for k in false_indices:
        action = [False] * len(state)
        action[k] = True
        list_of_actions.append(tuple(action))
    return list_of_actions


def generate_actions_and_init_q(states):
    dict_of_q_values = {}
    for state in states:
        actions = generate_actions_per_state(state)
        dict_of_q_values[state] = {}  # Initialize inner dict
        for action in actions:
            dict_of_q_values[state][action] = 0.0  # Init Q-value to 0.0 (float)
    return dict_of_q_values

#
# def init_q(states,actions):
#     q_values = {}
#     for state in states:
#         q_values[state] = {}
#         for action in actions:
#             q_values[state][action] = 0
#     return q_values

def get_optimal_action_according_to_q(q_values,state):
    best_action = min(q_values[state], key=q_values[state].get)
    return best_action

def choose_random_action(state):
    possible_actions = generate_actions_per_state(state)
    random_action = random.choice(possible_actions)
    return random_action

def update_q(q_values, state,action, next_state, cost,alpha):
    opt_q_next_state = q_values[next_state][get_optimal_action_according_to_q(q_values, next_state)]
    new_value = q_values[state][action] + alpha * (cost + opt_q_next_state - q_values[state][action])
    return new_value

def create_policy_from_q(q_values):
    policy = {}
    for state in q_values.keys():
        policy[state] = get_optimal_action_according_to_q(q_values, state)
    return policy

def q_learning(sub_section,opt_values):
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    delta = float('inf')
    job_names = list(env.action_space.keys())  # give me all that is not done
    states = generate_states(len(job_names))
    q_values = generate_actions_and_init_q(states)
    visits = {state: 0 for state in states}
    counter = 0
    max_norm = []
    abs_s0 = []


    while delta > EPSILON:
        old_q_values = copy.deepcopy(q_values)
        counter+=1
        # Initial state
        state = env.reset()[0]
        state = convert_sim_state_to_bool_state(state)

        for i in range(env.horizon):
            if random.random() < EXPLORATION:
                action = choose_random_action(state)
            else:
                action = get_optimal_action_according_to_q(q_values,state)

            action_sim = convert_action_to_sim_action_dict(job_names, action)
            # Step the environment
            next_state, cost, done, _, _ = env.step(action_sim)
            # print(next_state)

            alpha = 0.01

            if sub_section == "sub_section_1":
                alpha = 1 / visits[state]

            elif sub_section == "sub_section_2":
                alpha = 0.01

            elif sub_section == "sub_section_3":
                alpha = 10 / (100 + visits[state])

            next_state = convert_sim_state_to_bool_state(next_state)
            q_values[state][action] = update_q(q_values.copy(), state, action, next_state, cost,alpha)
            visits[state] += 1

            state = tuple(list(next_state).copy())

            # log data
            if done:
                break

        max_delta = 0.0
        for state in q_values:
            for action in q_values[state]:
                delta = abs(q_values[state][action] - old_q_values[state][action])
                if delta > max_delta:
                    max_delta = delta

        if max_delta < EPSILON:
            break
        else:
            delta = max_delta

        if counter % 100 == 0:

            policy = create_policy_from_q(q_values)
            values = mains2.evaluate_policy(policy, False)

            valid_states = [s for s in range(len(values)) if visits[s] > 0]
            if valid_states:
                max_norm.append(max(abs(values[s] - opt_values[s]) for s in valid_states))
                abs_s0.append(abs(values[-1] - opt_values[-1]))
                print(counter)
                print(delta)
            else:
                max_norm.append(0)  # Fallback in case nothing was visited yet
                print('i am here')


        if counter > 50000:
            break


    # Plot V(π_*) and V(π_Q) State Values
    plt.figure()
    plt.plot(opt_values, label="V(π_*)", marker='o', linewidth=1)
    plt.plot(values, label="V(π_Q)", marker='o', linewidth=1)
    plt.xlabel("State Index")
    plt.ylabel("Value")
    plt.title("V(π_*) and V(π_Q) State Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot V(π_TD)(s0) vs V(π_c)(s0)
    plt.figure()
    plt.plot(abs_s0, marker='o', linewidth=1)
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

opt_policy_values = mains2.evaluate_policy(generate_c_mu_policy(),False)
values = q_learning("sub_section_3",opt_policy_values)