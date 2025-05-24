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
from mains2 import COSTS

# Globals
EPSILON = 0.00000005
EXPLORATION = 1

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
        policy[state] = tuple(mains2.c_mu_action(state))
    return policy

def generate_states(num_jobs):
    return list(product([True, False], repeat=num_jobs))

def generate_cost_policy(states):
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
            dict_of_q_values[state][action] = 0  # Init Q-value to 0.0 (float)
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
    best_action = max(q_values[state], key=q_values[state].get)
    # if state == tuple([False] *5):
    #     print(best_action)
    return best_action

def choose_random_action(state):
    possible_actions = generate_actions_per_state(state)
    random_action = random.choice(possible_actions)
    return random_action

def update_q(q_values, state,action, next_state, cost,alpha):
    if state == tuple([False, False, True, True, True]):
        if action == tuple([True, False, False, False, False]):
            print('---')
            print('action:', action)
            print(f'same state: {state==next_state}')
            print(f'q value: {q_values[state][action]}')
            print(f'q value next: {q_values[next_state][get_optimal_action_according_to_q(q_values, next_state)]}')
            print(f' alpha: {alpha}')
            print(f'cost: {cost}')
            print(f'update: {(cost + q_values[next_state][get_optimal_action_according_to_q(q_values, next_state)] - q_values[state][action])}')

    opt_q_next_state = q_values[next_state][get_optimal_action_according_to_q(q_values, next_state)]

    update =  alpha * (cost + opt_q_next_state - q_values[state][action])
    return update

def create_policy_from_q(q_values):
    policy = {}
    for state in q_values.keys():
        policy[state] = get_optimal_action_according_to_q(q_values, state)
    return policy

def convert_from_state_to_index(states, available_states):
    indexes = []
    for s in available_states:
        if s in states:
            indexes.append(states.index(s))
    return indexes

def count_differences(list1, list2):
    # Ensure both lists are the same length
    if len(list1) != len(list2):
        raise ValueError("Lists must be the same length.")

    # Count differences
    return sum(1 for key in set(list1.keys()).union(list2.keys()) if list1.get(key) != list2.get(key))

def get_difference_indices(dict_list1, dict_list2):
    if len(dict_list1) != len(dict_list2):
        raise ValueError("Both lists must be of the same length.")

    diff_indices = []
    for i, (d1, d2) in enumerate(zip(dict_list1, dict_list2)):
        if dict_list1[d1] != dict_list2[d2]:
            diff_indices.append(i)
    return diff_indices

def q_learning(sub_section,opt_values):
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    delta = float('inf')
    job_names = list(env.action_space.keys())  # give me all that is not done
    states = generate_states(len(job_names))
    q_values = generate_actions_and_init_q(states)
    visits = {state: 0 for state in states}
    visits_per_action = generate_actions_and_init_q(states)
    counter = 0
    max_norm = []
    abs_s0 = []
    policy = create_policy_from_q(q_values)

    while True:
        pc = 0
    # while delta > EPSILON:
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
            # if state == tuple([False, False, True, True, True]):
            #     if action == tuple([True,False,False,False,False]):
            #         if state == convert_sim_state_to_bool_state(next_state):
            #             pc +=1
            #         else:
            #             pc-=1
            #         print(pc)

            visits[state] += 1
            visits_per_action[state][action] += 1


            if sub_section == "sub_section_1":
                alpha = 1 / visits[state]

            elif sub_section == "sub_section_2":
                alpha = 0.01

            elif sub_section == "sub_section_3":
                alpha = 10 / (100 + visits_per_action[state][action])
            else:
                print("alpha is not defined - crash!")

            next_state = convert_sim_state_to_bool_state(next_state)
            q_values_copy = copy.deepcopy(q_values)
            q_values[state][action] += update_q(q_values_copy, state, action, next_state, cost,alpha)


            # if state == (False,False,False,False,False):
            #     print(alpha)
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

        # if max_delta < EPSILON:
        #     break
        # else:
        #     delta = max_delta

        if counter % 100 == 0:
            old_policy = copy.deepcopy(policy)
            policy = create_policy_from_q(q_values)
            cmu_policy = generate_c_mu_policy()
            print(f'episode: {counter}')
            print("Differences between policy and optimal policy:",
                  count_differences(cmu_policy, policy))

            print("Differences indices:",
                  get_difference_indices(cmu_policy, policy))

            print("Qvalues for state:",q_values[tuple([False,False,True,True,True])])

            values = mains2.evaluate_policy(policy, False)

            valid_states = [s for s in states if visits[s] > 0]
            valid_states_indices = convert_from_state_to_index(states, valid_states)

            if valid_states:
                max_norm.append(max(abs(values[s] - opt_values[s]) for s in valid_states_indices))
                min_q = min(q_values[tuple([False]*5)].values())
                abs_s0.append(abs(min_q - opt_values[-1]))
            else:
                max_norm.append(0)  # Fallback in case nothing was visited yet
                print('i am here')


        if counter > 500000:
            print(q_values)
            print('-------')
            print(opt_values)
            break


    # Plot V(π_*) and V(π_Q) State Values
    # plt.figure()
    # plt.plot(opt_values, label="V(π_*)", marker='o', linewidth=1)
    # plt.plot(values, label="V(π_Q)", marker='o', linewidth=1)
    # plt.xlabel("State Index")
    # plt.ylabel("Value")
    # plt.title("V(π_*) and V(π_Q) State Values")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
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


    # if sub_section == "sub_section_1":
    #     alpha = "1/n_visits"
    # elif sub_section == "sub_section_2":
    #     alpha = "0.01"
    # elif sub_section == "sub_section_3":
    #     alpha = "10/(100+n_visits)"
    # plt.plot(max_norm, label=f"{sub_section}, alpha: {alpha}")
    # plt.title(r"$\|V^{\pi_c} - \hat{V}_{TD}\|_\infty$ over Episodes")
    # plt.xlabel("Episode")
    # plt.ylabel("Infinity Norm Error")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

opt_policy_values = mains2.evaluate_policy(generate_c_mu_policy(),False)
values = q_learning("sub_section_2",opt_policy_values)