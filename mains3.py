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
from mains2 import generate_states, generate_cost_policy,policy_iteration
# Globals
EPSILON = 0.00000005


# Global variables
base_path ='./' # '../../Desktop/'
domain_file = base_path + 'jobs_domain.rddl'
instance_file = base_path + 'jobs_instance.rddl'

"""""""""""""""""""""""""""
Question 2   -   Learning 1
"""""""""""""""""""""""""""

def convert_action_to_sim_action_dict(job_names,bool_action):
    # Converts actions from internal format to simulator format
    action_dict = {job: val for job, val in zip(job_names, bool_action)}
    return action_dict

def convert_sim_state_to_bool_state(sim_state):
    # Converts state from simulator format to internal format
    was_done = {}
    for key in sim_state.keys():
        match = re.match(r'was_done___j(\d+)', key)
        if match:
            index = int(match.group(1)) - 1  # convert j1-based to 0-based
            was_done[index] = sim_state[key]
    return tuple([was_done[i] for i in sorted(was_done)])

def policy_evaluation(sub_section,policy_name='c'):
    # Do Temporal Difference Policy Evaluation
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    delta = float('inf')
    job_names = list(env.action_space.keys())  # give me all that is not done
    states = generate_states()
    values = {state: 0 for state in states}
    visits = {state: 0 for state in states}
    policy_cost = generate_cost_policy(states)

    if policy_name == 'c':
        policy = policy_cost
    elif policy_name == 'cmu':
        policy, V_star = policy_iteration(initial_policy=policy_cost, graph= False)
    else:
        print('print not valid input')

    max_norm = []
    v_pi_c_raw = mains2.evaluate_policy(policy, False)
    v_pi_c = dict(zip(states, v_pi_c_raw))

    v_pi_c_so = v_pi_c[tuple([False]*len(states[0]))] # Scalar
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
            next_state = convert_sim_state_to_bool_state(next_state)

            visits[state] += 1

            if sub_section == "sub_section_1":
                alpha = 1/visits[state]

            elif sub_section == "sub_section_2":
                alpha = 0.01

            elif sub_section == "sub_section_3":
                alpha = 10/(100 + visits[state])
            else:
                print("error - function crash!")

            current_value = values[state]
            values[state] = current_value + alpha * (cost + values[next_state] - current_value)

            state = tuple(list(next_state).copy())

            if done:
                break

        if counter>20000:
            break

        # The norm is being calculated only from states that where visited (other states values are not participated)
        valid_states = [s for s in values if visits[s] > 0 and s in v_pi_c]
        if valid_states:
            max_norm.append(max(abs(v_pi_c[s] - values[s]) for s in valid_states))
        else:
            max_norm.append(0)  # Fallback in case nothing was visited yet
            print('Fallback - max norm added with zero')

        v_TD_so.append(values[tuple([False] * len(states[0]))])
        delta = max(abs(old_values[s] - values[s]) for s in states)
        if counter % 1000 == 0:
            print(counter)
            print(delta)
        if delta <= EPSILON:
            print(f"Stop: {delta}")


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

    if sub_section == "sub_section_1":
        alpha = "1/n_visits"
    elif sub_section == "sub_section_2":
        alpha = "0.01"
    elif sub_section == "sub_section_3":
        alpha = "10/(100+n_visits)"

    # Plot Norm (V(π_TD) - V(π_c)) vs Episode
    plt.plot(max_norm)
    plt.title(r"$\|V^{\pi_c} - \hat{V}_{TD}\|_\infty$ over Episodes, alpha: "+ alpha, fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Infinity Norm Error", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot V(π_TD)(s0) - V(π_c)(s0) vs Episode
    delta_s0 = abs(v_TD_so - v_pi_c_so)
    plt.figure()
    plt.plot(delta_s0)
    plt.xlabel("Episode",fontsize=12)
    plt.ylabel("Absolute Error",fontsize=12)
    plt.title(f"|V(π_TD)(S0) - V(π_c)(S0)| VS Episode, alpha: {alpha}",fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return max_norm

if __name__ == '__main__':

    # sub_section_1 - alpha = 1/no.of visits to Sn
    # sub_section_2 - alpha = 0.01
    # sub_section_3 - alpha = 10/(100 + no.of visits to Sn)

    random.seed(0)
    max_norm_1 = policy_evaluation("sub_section_1")
    max_norm_2 = policy_evaluation("sub_section_2")
    max_norm_3 = policy_evaluation("sub_section_3")








