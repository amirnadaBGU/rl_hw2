import itertools
import random
import pyRDDLGym
import numpy as np
import matplotlib
from itertools import product, cycle

matplotlib.use('TkAgg')
from itertools import product
from matplotlib import pyplot as plt
import math


# Global variables
base_path ='./' # '../../Desktop/'
domain_file = base_path + 'jobs_domain.rddl'
instance_file = base_path + 'jobs_instance.rddl'

# GLOBALS
COSTS = [-1, -4]
MUS = np.array([0.5, 1])
EPSILON = 0.05

def generate_states():
    return list(product([True, False], repeat=2))

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
            if COSTS[i] < max_cost:
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
        # value = get_total_cost(state)
        value = 0
        values.append(value)
    return values

def evaluate_policy(policy, graph = True):
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

    if graph:
        plot_value_history(value_history, states)

    return values

def plot_value_history(value_history, states, max_legend_cols=4):
    value_history = np.array(value_history)
    assert value_history.shape[1] == len(states), "Mismatch between value history and number of states"

    plt.figure(figsize=(20, 10))

    # Use tab20 for distinct colors
    base_colors = plt.get_cmap("tab20").colors  # 20 distinct RGB colors
    colors = base_colors[:16] + base_colors[16:]  # Use full 20

    # 8 unique line styles
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (1, 2))]

    # Generate (color, linestyle) pairs
    combinations = [(color, style) for style in line_styles for color in colors]
    assert len(combinations) >= len(states), "Not enough unique combinations"

    for i, state in enumerate(states):
        label = ''.join(['t' if x else 'f' for x in state])
        color, style = combinations[i]
        plt.plot(value_history[:, i], label=label, color=color, linestyle=style, linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("V(state)")
    plt.title("Value Function Convergence")
    plt.grid(True)
    plt.legend(ncol=max_legend_cols, fontsize='small', loc='upper left', bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.show()

def policy_iteration(initial_policy, graph):

    def find_best_action(states, index, values):
        state = states[index]
        false_indices = [i for i, val in enumerate(state) if not val]
        max_Q = -float('inf')
        best_action = [False] * len(state)

        if not false_indices:
            return best_action

        for i in false_indices:
            action = [False] * len(state)
            action[i] = True

            q = calc_value(states, index, action, values)

            if q > max_Q:
                max_Q = q
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

    if graph:
        # Plot value function convergence
        value_history = np.array(value_history)
        plt.figure()
        # for i in range(value_history.shape[1]):
        #     plt.plot(value_history[:, i], label=f"State {i}")

        plt.plot(value_history[:, -1])

        plt.xlabel("Iteration")
        plt.ylabel("V(S0)")
        plt.title("Value of S0 Function Convergence During Policy Iteration")
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return policy, values

def c_mu_action(state):
    best_score = float('inf')
    best_job = None
    for i, done in enumerate(state):
        if not done:
            score = COSTS[i] * MUS[i]
            if score < best_score:
                best_score = score
                best_job = i
    return [i == best_job for i in range(len(state))]

def generate_c_mu_policy():
    env = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    job_names = list(env.action_space.keys())  # give me all that is not done
    states = generate_states()
    policy ={}
    for state in states:
        policy[state] = c_mu_action(state)
    return policy

# Compare the optimal policy π∗ obtained using policy iteration to the cμ law.
# Also plot V π∗ vs. V πc .

def comparison_optimal_policy_and_cm():
    print()


if __name__ == '__main__':
    section_0 = False # Draft
    section_3 = False # Corresponds to question planning 3
    section_4 = False # Corresponds to question planning 4
    section_5 = True # Corresponds to question planning 5

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

            was_done = [next_state['was_done___j1'], next_state['was_done___j2']]

            print(was_done)
            print(cost)

    if section_3 or section_5:
        states = generate_states()
        print(f"{len(set(states))}: different states")
        for k,state in enumerate(states):
                print(k)
                print(state)
                print('-----')
        random.seed(42)
        cost_policy = generate_cost_policy()

        if section_5:
            graph = False
        else:
            graph = True

        V_cost = evaluate_policy(cost_policy, graph)

    if section_4 or section_5:
        cost_policy = generate_cost_policy()

        if section_5:
            graph = False
        else:
            graph = True
        optimal_policy, V_star = policy_iteration(initial_policy=cost_policy, graph= graph)
        print(optimal_policy)

    if section_5:
        states = generate_states()
        c_mu_policy = generate_c_mu_policy()
        count_of_different = 0
        for state in states:
            optimal = optimal_policy[tuple(state)]
            cmu = c_mu_policy[tuple(state)]
            if optimal != cmu:
                print(f"Difference at state {state}:")
                print(f"  π*:  {optimal}")
                print(f"  cμ:  {cmu}")
                count_of_different += 1

        print(f'number of actions that are different between optimal policy and cμ law: {count_of_different}')

        V_c_mu_policy = evaluate_policy(c_mu_policy, False)

        # Generate state labels
        states = generate_states()
        state_labels = [','.join(['T' if b else 'F' for b in s]) for s in states]

        # Plot
        plt.figure()
        plt.plot(V_star, label="V(π*) - Optimal Policy", marker='o', linewidth=1)
        plt.plot(V_c_mu_policy, label="V(π_cμ) - cμ Policy", marker='x', linewidth=1)
        plt.xlabel("State", fontsize= 10)
        plt.ylabel("Value", fontsize= 10)
        plt.title("Comparison: V(π*) vs V(π_cμ)")
        plt.xticks(ticks=range(len(states)), labels=state_labels, rotation=90, fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot
        plt.figure()
        plt.plot(V_star, label="V(π*) - Optimal Policy", marker='o', linewidth=1)
        plt.plot(V_cost, label="V(π_c) - Cost-based Policy", marker='x', linewidth=1)
        plt.xlabel("State", fontsize= 10)
        plt.ylabel("Value", fontsize= 10)
        plt.title("Comparison: V(π*) vs V(π_c)")
        plt.xticks(ticks=range(len(states)), labels=state_labels, rotation=90, fontsize=8)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


