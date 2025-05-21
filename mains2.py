import itertools

import pyRDDLGym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import math


# Global variables
base_path ='./' # '../../Desktop/'
domain_file = base_path + 'jobs_domain.rddl'
instance_file = base_path + 'jobs_instance.rddl'

# GLOBALS


def policy_evaluation(policy):
    """
    policy: dict mapping each state (tuple of 0/1) to a job index (1..N)
    mu: list of success probabilities per job
    c: list of costs per job
    """
    import numpy as np
    from itertools import product

    env = pyRDDLGym.make(domain=domain_file, instance=instance_file)

    # Extract all job names
    jobs = list(env.nonfluents['mu'].keys())

    # Extract mu and cost values
    mu = [env.nonfluents['mu'][job] for job in jobs]
    c = [env.nonfluents['cost'][job] for job in jobs]

    n = len(mu)
    states = list(product([0, 1], repeat=n))  # 0 = unfinished, 1 = finished
    state_to_index = {s: i for i, s in enumerate(states)}
    V = np.zeros(len(states))
    A = np.zeros((len(states), len(states)))
    b = np.zeros(len(states))

    for i, s in enumerate(states):
        if all(s):  # terminal state
            A[i, i] = 1
            b[i] = 0
            continue
        action = policy[s]  # job to serve
        job_index = action - 1  # 0-based
        cost = sum(c[j] for j in range(n) if s[j] == 0)
        mu_a = mu[job_index]

        s_prime = list(s)
        s_prime[job_index] = 1
        idx_prime = state_to_index[tuple(s_prime)]

        A[i, i] = 1 - (1 - mu_a)  # stays in same state with 1 - mu
        A[i, idx_prime] = -mu_a
        b[i] = cost

    V = np.linalg.solve(A, b)
    return V


if __name__ == '__main__':
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



    #full_plot_run(do_single_run_random_policy)
    #full_plot_run(do_single_run_greedy_policy)
