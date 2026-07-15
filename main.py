import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict 
from Environments.control_gated import ControlGatedMDP
from Environments.road import Road
from Environments.two_room import TwoRoom
from agent import Agent
from metrics import empowerment_with_memory, plasticity_with_memory
from tqdm import tqdm

# Configuration 
N_TRIALS = 100
ITERATIONS = 100
env = ControlGatedMDP()

# Arrays to aggregate metrics across all trials
all_v_histories = []
all_plasticity_metrics = []
all_empowerment_metrics = []

print(f"Starting {N_TRIALS} independent trials...")

for trial in range(N_TRIALS):
    print(f"\n--- Trial {trial + 1}/{N_TRIALS} ---")
    
    # Re-initialize agent every trial to reset 
    agent = Agent(
        n_states=env.n_states, 
        n_actions=env.n_actions, 
        gamma=0.9, 
        transition_function=env.transition_function,
        initial_state_distribution=env.init_distribution_function,
    )

    # Initialize data trackers for this trial
    v_history = []
    plasticity_metrics = []
    empowerment_metrics = []

    # Wrapper matching functional initial state array distribution
    initial_distribution_array = np.array([env.init_distribution_function(s) for s in range(env.n_states)])

    # Capture initial baseline before optimization iterations begin
    plasticity_metrics.append(plasticity_with_memory(agent.get_policy(), initial_distribution_array, env.transition_function, agent.n_states, agent.n_actions)[0])
    empowerment_metrics.append(empowerment_with_memory(agent.get_policy(), initial_distribution_array, env.transition_function, agent.n_states, agent.n_actions)[0])

    # Optimization Iteration Loop
    for iter in tqdm(range(ITERATIONS), desc=f"Training Agent (Trial {trial+1})"):

        # Update Agent
        agent.update()
        
        # Track distributions
        v_history.append(np.copy(agent.V_table))
        
        current_policy = agent.get_policy()
        plasticity_metrics.append(plasticity_with_memory(current_policy, initial_distribution_array, env.transition_function, agent.n_states, agent.n_actions)[0])
        empowerment_metrics.append(empowerment_with_memory(current_policy, initial_distribution_array, env.transition_function, agent.n_states, agent.n_actions)[0])

    # Save data from this trial
    all_v_histories.append(v_history)
    all_plasticity_metrics.append(plasticity_metrics)
    all_empowerment_metrics.append(empowerment_metrics)

# Convert lists to numpy arrays for calculation
all_v_histories = np.array(all_v_histories)
all_plasticity_metrics = np.array(all_plasticity_metrics)    
all_empowerment_metrics = np.array(all_empowerment_metrics) 

# Calculate means and standard deviations
v_mean = np.mean(all_v_histories, axis=0)
v_std = np.std(all_v_histories, axis=0)

plasticity_mean = np.mean(all_plasticity_metrics, axis=0)
plasticity_std = np.std(all_plasticity_metrics, axis=0)

empowerment_mean = np.mean(all_empowerment_metrics, axis=0)
empowerment_std = np.std(all_empowerment_metrics, axis=0)


# Plotting 
history_states = list(range(env.n_states)) + [-1]
history_actions = list(range(env.n_actions)) + [-1]

x_values = np.arange(ITERATIONS + 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Value Function Plot
ax1 = axes[0]

# Pre-grouping data structures
unique_lines = defaultdict(list)
unique_stds = {}

for sh in history_states:
    for ah in history_actions:
        if (sh == -1 and ah != -1) or (ah == -1 and sh != -1):
            continue
            
        y_mean_arr = v_mean[:, sh, ah]
        y_std_arr = v_std[:, sh, ah]
        
        if np.all(y_mean_arr == 0):
            continue
            
        lbl = "None" if sh == -1 else f"({sh},{ah})"
        
        key = tuple(y_mean_arr)
        unique_lines[key].append(lbl)
        unique_stds[key] = y_std_arr

# Plotting the aggregated lines
for y_mean_tuple, states_list in unique_lines.items():
    y_mean_plot = np.array(y_mean_tuple)
    y_std_plot = unique_stds[y_mean_tuple]
    
    # Dynamic legend label truncation so it doesn't spill over your graph
    if len(states_list) > 3:
        combined_label = f"V: {', '.join(states_list[:3])} ... (+{len(states_list)-3} more)"
    else:
        combined_label = f"V: {', '.join(states_list)}"
        
    # Plot single mean line for this unique array group
    line, = ax1.plot(x_values[1:], y_mean_plot, marker='o', alpha=0.8, label=combined_label)
    
    # Plot variance shading matching the unique group
    ax1.fill_between(x_values[1:], y_mean_plot - y_std_plot, y_mean_plot + y_std_plot, color=line.get_color(), alpha=0.15)

ax1.set_title(f'Mean Value Convergence over {N_TRIALS} Trials', fontsize=12)
ax1.set_xlabel('Outer Iterations', fontsize=11)
ax1.set_ylabel('Value Score (Mean ± STD)', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='best', fontsize='small')

# Plasticity and Empowerment Plot
ax2 = axes[1]

# Plasticity
line_p, = ax2.plot(x_values, plasticity_mean, color='tab:orange', linewidth=2, label='Plasticity')
ax2.fill_between(x_values, plasticity_mean - plasticity_std, plasticity_mean + plasticity_std, color='tab:orange', alpha=0.2)

# Empowerment
line_e, = ax2.plot(x_values, empowerment_mean, color='tab:green', linewidth=2, label='Empowerment')
ax2.fill_between(x_values, empowerment_mean - empowerment_std, empowerment_mean + empowerment_std, color='tab:green', alpha=0.2)

ax2.set_title(f'Mean Plasticity and Empowerment over {N_TRIALS} Trials', fontsize=12)
ax2.set_xlabel('Outer Iterations', fontsize=11)
ax2.set_ylabel('Bits (Mean ± STD)', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='best')

plt.tight_layout()
plt.show()