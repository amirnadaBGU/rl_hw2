import copy
import random


import pyRDDLGym
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mains2 import generate_c_mu_policy,generate_states, evaluate_policy
from mains3 import convert_action_to_sim_action_dict,convert_sim_state_to_bool_state

"""""""""""""""""""""""""""
Question 2   - Learning 2,3
"""""""""""""""""""""""""""

# Globals
EPSILON = 0.0005
NUMBER_OF_ITERATIONS = 100000

# Global Paths
base_path ='./' # '../../Desktop/'
domain_file = base_path + 'jobs_domain.rddl'
instance_file = base_path + 'jobs_instance.rddl'


def generate_actions_per_state(state):
    # Return list of avaliable actions per state
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
    # Initializing q matrix with zeros
    dict_of_q_values = {}
    for state in states:
        actions = generate_actions_per_state(state)
        dict_of_q_values[state] = {}  # Initialize inner dict
        for action in actions:
            dict_of_q_values[state][action] = 0  # Init Q-value to 0.0 (float)
    return dict_of_q_values

def get_optimal_action_according_to_q(q_values,state):
    # Choose greedy action according to q values
    best_action = max(q_values[state], key=q_values[state].get)
    return best_action

def choose_random_action(state):
    # Random policy action
    possible_actions = generate_actions_per_state(state)
    random_action = random.choice(possible_actions)
    return random_action

def update_q(q_values, state,action, next_state, cost,alpha):
    # Update entry in  values matrix according to the q learning formula
    opt_q_next_state = q_values[next_state][get_optimal_action_according_to_q(q_values, next_state)]
    current_q = q_values[state][action]
    new_value = q_values[state][action] + alpha * (cost + opt_q_next_state - current_q)
    return new_value

def create_policy_from_q(q_values):
    # Creates policy by acting greedly according to Q
    policy = {}
    for state in q_values.keys():
        policy[state] = get_optimal_action_according_to_q(q_values, state)
    return policy

def convert_from_state_to_index(states, available_states):
    # Function that return the location of a state in the state dictionary
    indexes = []
    for s in available_states:
        if s in states:
            indexes.append(states.index(s))
    return indexes

def count_differences(list1, list2):
    # Function used for debug policy convergence - Not in use
    if len(list1) != len(list2):
        raise ValueError("Lists must be the same length.")

    # Count differences
    return sum(1 for key in set(list1.keys()).union(list2.keys()) if list1.get(key) != list2.get(key))

def q_learning(sub_section,opt_values, exploration):
    # function that do q_learning procedure

    env = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    delta = float('inf')
    job_names = list(env.action_space.keys())  # give me all that is not done
    states = generate_states(len(job_names))
    q_values = generate_actions_and_init_q(states)
    visits = {state: 0 for state in states}
    counter = 0
    max_norm = []
    abs_s0 = []

    # Episodes iteration:
    while delta > EPSILON:
        old_q_values = copy.deepcopy(q_values)
        counter+=1
        state = env.reset()[0]
        state = convert_sim_state_to_bool_state(state)

        for i in range(env.horizon):
            # Epsilon greedy method
            rnd = random.random()
            if rnd < exploration:
                action = choose_random_action(state)
            else:
                action = get_optimal_action_according_to_q(q_values,state)

            action_sim = convert_action_to_sim_action_dict(job_names, action)
            next_state, cost, done, _, _ = env.step(action_sim)

            # Update visits
            visits[state] += 1

            # Determine update rate
            if sub_section == "sub_section_1":
                alpha = 1 / visits[state]

            elif sub_section == "sub_section_2":
                alpha = 0.01

            elif sub_section == "sub_section_3":
                alpha = 10 / (100 + visits[state])
            else:
                print("Error - no appropriate sub-section")

            next_state = convert_sim_state_to_bool_state(next_state)

            # Do single q_value update
            q_values[state][action] = update_q(copy.deepcopy(q_values), state, action, next_state, cost,alpha)


            state = tuple(list(next_state).copy())

            if done:
                break

        # Calculating stopping criteria
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

        # Printing and data storage every 100 episodes
        if counter % 100 == 0:
            valid_states = [s for s in states if visits[s] > 0]
            if valid_states:
                max_norm.append(
                    max(
                        abs(max(q_values[s].values()) - opt_values[convert_from_state_to_index(states, [s])[0]])
                        for s in valid_states
                    )
                )
                max_q_s0 = max(q_values[tuple([False]*5)].values())
                abs_s0.append(abs(max_q_s0 - opt_values[-1]))
                print(counter)
                print(delta)
            else:
                max_norm.append(0)  # Fallback in case nothing was visited yet

        # Hard Stopping criteria
        if counter > NUMBER_OF_ITERATIONS:
            break

    return abs_s0, max_norm


def plot_function3(abs_s0_1, max_norm_1,abs_s0_2, max_norm_2, abs_s0_3, max_norm_3):
    episodes = [100 * (i + 1) for i in range(len(abs_s0_1))]

    plt.plot(episodes, max_norm_1, label="1/n_visits")
    plt.plot(episodes, max_norm_2, label="0.01")
    plt.plot(episodes, max_norm_3, label="10/(100+n_visits)")
    plt.title(r"$\left\| V^{\ast} - \hat{V}^{\pi_{\hat{Q}}} \right\|_{\infty}$ over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Infinity Norm Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(episodes, abs_s0_1, label="1/n_visits")
    plt.plot(episodes, abs_s0_2, label="0.01")
    plt.plot(episodes, abs_s0_3, label="10/(100+n_visits)")
    plt.xlabel("Episode")
    plt.ylabel("Absolute Error")
    plt.title(r"$\left| V^{\ast}(s_0) - \max_a \hat{Q}(s_0, a) \right|$ vs Episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_function2(abs_s0_1, max_norm_1,abs_s0_2, max_norm_2,exp1,exp2):
    episodes = [100 * (i + 1) for i in range(len(abs_s0_1))]

    plt.plot(episodes, max_norm_1, label=f"Exploration Rate: {exp1}")
    plt.plot(episodes, max_norm_2, label=f"Exploration Rate: {exp2}")
    plt.title(r"$\left\| V^{\ast} - \hat{V}^{\pi_{\hat{Q}}} \right\|_{\infty}$ over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Infinity Norm Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(episodes, abs_s0_1, label=f"Exploration Rate: {exp1}")
    plt.plot(episodes, abs_s0_2, label=f"Exploration Rate: {exp2}")
    plt.xlabel("Episode")
    plt.ylabel("Absolute Error")
    plt.title(r"$\left| V^{\ast}(s_0) - \max_a \hat{Q}(s_0, a) \right|$ vs Episode")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':

    # sub_section_1 - alpha = 1/no.of visits to Sn
    # sub_section_2 - alpha = 0.01
    # sub_section_3 - alpha = 10/(100 + no.of visits to Sn)
    s4 = True
    s5 = True
    if s4:
        random.seed(0)
        opt_policy_values = evaluate_policy(generate_c_mu_policy(), False)
        abs_s0_1, max_norm_1 = q_learning("sub_section_1", opt_policy_values, 0.1)
        abs_s0_2, max_norm_2 = q_learning("sub_section_2", opt_policy_values, 0.1)
        abs_s0_3, max_norm_3 = q_learning("sub_section_3", opt_policy_values, 0.1)
        plot_function3(abs_s0_1, max_norm_1, abs_s0_2, max_norm_2, abs_s0_3, max_norm_3)
    if s5:
        random.seed(0)
        opt_policy_values = evaluate_policy(generate_c_mu_policy(), False)
        abs_s0_ep_01, max_norm_ep_01 = q_learning("sub_section_3", opt_policy_values, exploration=0.1)
        abs_s0_ep_001, max_norm_ep_001 = q_learning("sub_section_3", opt_policy_values, exploration=0.01)
        plot_function2(abs_s0_ep_01, max_norm_ep_01, abs_s0_ep_001, max_norm_ep_001, 0.1, 0.01)