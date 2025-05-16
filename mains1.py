import pyRDDLGym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# Global variables
DEBUG = False
base_path = './'
domain_file = base_path + 'bandit_domain.rddl'
instance_file = base_path + 'bandit_instance.rddl'

# Parse horizon from instance file
HORIZON = 100 #TODO: horizon is not equal here and in instance file
MEAN_BEST_REWARD_PER_ACTION = 0.5




def random_policy():
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

        # print(f"Step {i+1}: Rolled {chosen_arm}, Reward: {reward}")
        rewards.append(reward)
    return rewards

def plot_rew_reg(rew_reg,headline):
    cumulative_rew_reg = np.cumsum(rew_reg)
    avg_total_rew_reg = np.mean(rew_reg)
    print(f"Policy: {headline}, average quantity over {HORIZON} steps = {avg_total_rew_reg:.3f}")
    plt.plot(cumulative_rew_reg, marker='.', linewidth=0.5, markersize=2)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title(f'Policy: {headline} - Rewards per Step')
    plt.grid(True)
    plt.show()

def full_plot_run(policy):
    random_rewards_all_runs = []
    random_regrets_all_runs = []
    for k in range(20):
        # Collect rewards
        rewards_per_run = policy()

        # Collect regrets
        regrets_per_run = MEAN_BEST_REWARD_PER_ACTION * np.ones(len(rewards_per_run))-rewards_per_run

        random_rewards_all_runs.append(rewards_per_run)
        random_regrets_all_runs.append(regrets_per_run)

    random_rewards_all_runs = np.array(random_rewards_all_runs)
    random_regrets_all_runs = np.array(random_regrets_all_runs)

    average_rewards = np.mean(random_rewards_all_runs, axis=0)
    average_regrets = np.mean(random_regrets_all_runs, axis=0)

    plot_rew_reg(average_rewards, headline="Random Policy Rewards")
    plot_rew_reg(average_regrets, headline="Random Policy Regrets")


if __name__ == '__main__':
    full_plot_run(random_policy)