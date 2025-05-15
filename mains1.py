import pyRDDLGym
import numpy as np
from matplotlib import pyplot as plt

# Global variables
DEBUG = False
base_path = './'
domain_file = base_path + 'bandit_domain.rddl'
instance_file = base_path + 'bandit_instance.rddl'

# Parse horizon from instance file
HORIZON = 20000 #TODO: horizon is not equal here and in instance file

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
    chosen_arm = "roll___a1"
    # Create an action dict with all False, except one arm
    action = {arm: False for arm in arm_names}
    action[chosen_arm] = True

    # Step the environment
    next_state, reward, done, _, _ = myEnv.step(action)

    # print(f"Step {i+1}: Rolled {chosen_arm}, Reward: {reward}")
    rewards.append(reward)

print(np.mean(rewards))
# Optional: Plot the rewards
plt.plot(rewards, marker='o')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('3 Random Rolls')
plt.grid(True)
plt.show()
