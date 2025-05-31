import pyRDDLGym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import math


# Global variables
DEBUG = False
base_path = './'
domain_file = base_path + 'bandit_domain.rddl'
instance_file = base_path + 'bandit_instance.rddl'

# GLOBALS
HORIZON = 20000 # suppose to be 20000
N_LOOPS = 20 # suppose to be 20
MEAN_BEST_REWARD_PER_ACTION = 100.0/101.0 # for regret calculations

def do_single_run_ucb1_policy():
    """
        function that do full simulation with greedy policy according to instructions - Horizon arm choices sequence
        ______
        return:
        ______
            rewards - list of rewards. Shape: [r,r,r...] - length HORIZON
    """
    # Create the environment
    myEnv = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    # Extract arm names from action space
    arm_names = list(myEnv.action_space.keys())
    # Start the environment
    myEnv.reset()
    # Construct reward poe arm array:
    reward_per_arm = np.zeros(len(arm_names))
    rewards = []
    num_of_pulls_per_arm = np.zeros(len(arm_names))

    # First, try each arm once (could also be obtained by UCB1 automatically)
    for i in range(len(arm_names)):
        # Choose a arm in a sequntial order
        chosen_arm = arm_names[i]

        # Create an action dict with all False, except one arm
        action = {arm: False for arm in arm_names}
        action[chosen_arm] = True

        # Step the environment
        next_state, reward, done, _, _ = myEnv.step(action)

        # Update reward per arm
        reward_per_arm[i] += reward
        num_of_pulls_per_arm[i] += 1

        # Update rewards
        rewards.append(reward)

    # Secondly act greedily with respect to UCB1
    for i in range(HORIZON-len(arm_names)):
        num_of_pulls = sum(num_of_pulls_per_arm)

        # UCB formula applied to each arm
        ucb_cumulative_reward = [
            (reward_per_arm[i] / num_of_pulls_per_arm[i]) +
            math.sqrt((2 * math.log(num_of_pulls)) / num_of_pulls_per_arm[i])
            for i in range(len(reward_per_arm))
        ]

        best_arm = np.argmax(ucb_cumulative_reward)

        # Choose a arm in a sequntial order
        chosen_arm = arm_names[best_arm]

        # Create an action dict with all False, except one arm
        action = {arm: False for arm in arm_names}
        action[chosen_arm] = True

        # Step the environment
        next_state, reward, done, _, _ = myEnv.step(action)

        # Update reward per arm
        reward_per_arm[best_arm] += reward
        num_of_pulls_per_arm[best_arm] += 1

        # Update rewards
        rewards.append(reward)

    return rewards

def do_single_run_greedy_policy():
    """
    function that do full simulation with greedy policy according to instructions - Horizon arm choices sequence
    ______
    return:
    ______
        rewards - list of rewards. Shape: [r,r,r...] - length HORIZON
    """
    # Create the environment
    myEnv = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    # Extract arm names from action space
    arm_names = list(myEnv.action_space.keys())
    # Start the environment
    myEnv.reset()
    # Construct reward poe arm array:
    reward_per_arm = np.zeros(len(arm_names))
    rewards = []
    for i in range(len(arm_names)):
        for j in range(100):
            # Choose a arm in a sequntial order
            chosen_arm = arm_names[i]

            # Create an action dict with all False, except one arm
            action = {arm: False for arm in arm_names}
            action[chosen_arm] = True

            # Step the environment
            next_state, reward, done, _, _ = myEnv.step(action)

            # Update reward per arm
            reward_per_arm[i] += reward

            # Update rewards
            rewards.append(reward)
    best_arm = np.argmax(reward_per_arm)
    chosen_arm = arm_names[best_arm]

    # Create an action dict with all False, except one arm
    action = {arm: False for arm in arm_names}
    action[chosen_arm] = True
    for i in range(10000):

        # Step the environment
        next_state, reward, done, _, _ = myEnv.step(action)

        # Update rewards
        rewards.append(reward)


    return rewards

def do_single_run_random_policy():
    """
    function that do full simulation with random policy - Horizon arm choices sequence
    ______
    return:
    ______
        rewards - list of rewards. Shape: [r,r,r...] - length HORIZON
    """
    # Create the environment
    myEnv = pyRDDLGym.make(domain=domain_file, instance=instance_file)
    # Extract arm names from action space
    arm_names = list(myEnv.action_space.keys())
    # Start the environment
    myEnv.reset()
    rewards = []
    for i in range(HORIZON):
        # Choose a random arm
        chosen_arm = np.random.choice(arm_names)
        # Create an action dict with all False, except one arm
        action = {arm: False for arm in arm_names}
        action[chosen_arm] = True

        # Step the environment
        next_state, reward, done, _, _ = myEnv.step(action)

        # Update rewards
        rewards.append(reward)
    return rewards

def plot_rew_reg(rew_reg,label,rew_reg_str='regret'):
    """
    function that plots a single graph of cumulative np array. can be rewards or regrest
    ______
    input:
        rew_reg - average rewards/regret np array. Shaoe: [ar,ar,ar,....ar] - HORIZON
        label - string headline provided by the user - method
        rew_reg_str - reward or regret plot
    """
    cumulative_rew_reg = np.cumsum(rew_reg)
    avg_total_rew_reg = np.mean(rew_reg)
    print(f"Policy: {label}, average quantity over {HORIZON} steps = {avg_total_rew_reg:.3f}")
    plt.plot(cumulative_rew_reg, marker='.', linewidth=0.5, markersize=2,label=label)
    plt.xlabel('Step',fontsize=22)
    if rew_reg_str == 'regret':
        plt.ylabel('Regret',fontsize=22)
    else:
        plt.ylabel('Reward',fontsize=22)
    plt.title(f'Three Policies - Average Cumulative Regret VS Step (20 experiments each)', fontsize=30)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

def full_plot_run(policy):
    """
    function that calls 20 time input function policy
    collects rewards, average them per step and plot them using plot_rew_reg

    intput:
        policy - function that runs a simulation with specific policy
    """
    rewards_all_runs = []
    regrets_all_runs = []
    for k in range(N_LOOPS):
        print(f"Start {k+1} Run!")
        # Collect rewards - Shape [r,r,r....] - length horizon
        rewards_per_run = policy()

        # Collect regrets - Shape [r,r,r....] - length horizon
        regrets_per_run = MEAN_BEST_REWARD_PER_ACTION * np.ones(len(rewards_per_run))-rewards_per_run

        # Summing - Shape [[r,r,r....],[r,r,r....]] - [[length horizon],] - length 20
        rewards_all_runs.append(rewards_per_run)
        regrets_all_runs.append(regrets_per_run)
        print(f"Finish {k + 1} Run!")
    # Shaing as numpy array from list
    rewards_all_runs = np.array(rewards_all_runs)
    regrets_all_runs = np.array(regrets_all_runs)

    # Average value per step - Shape [av_r,av_r.....] - length Horizon
    average_rewards = np.mean(rewards_all_runs, axis=0)
    average_regrets = np.mean(regrets_all_runs, axis=0)

    # Do ploting
    if policy is do_single_run_random_policy:
        headline = "Random Policy"
    elif policy is do_single_run_greedy_policy:
        headline = "Greedy Policy"
    elif policy is do_single_run_ucb1_policy:
        headline = "UCB1 Policy"

    # plot_rew_reg(average_rewards, headline=f"{headline}")
    plot_rew_reg(average_regrets, label=f"{headline}")


if __name__ == '__main__':
    full_plot_run(do_single_run_random_policy)
    full_plot_run(do_single_run_greedy_policy)
    full_plot_run(do_single_run_ucb1_policy)
    plt.show()
