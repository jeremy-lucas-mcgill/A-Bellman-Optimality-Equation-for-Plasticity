import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

class TwoRoom:
    def __init__(self, maxsteps=10):
        # States and Actions
        self.n_states = 35
        self.n_actions = 4

        # Max steps
        self.max_steps = maxsteps

        # Start State
        self.start_state = 15

        # Steps tracker
        self.steps = 0

        # Initial State Distribution
        self.initial_state_distribution = np.zeros(self.n_states)
        self.initial_state_distribution[self.start_state] = 1

        # Initial State Distribution Function
        self.init_distribution_function = lambda s: self.initial_state_distribution[s]

        # Transition Function
        self.transition_function = lambda s, a, s1=None: self.transition(s,a, s1)
        
    
    def reset(self):
        # Reset steps and state
        self.steps = 0
        self.state = self.start_state
        return self.state
    
    def step(self, action):
        # Increment Steps
        self.steps += 1

        # Get next state probabilities
        next_state_probs = self.transition_function(self.state, action)

        # Sample next state
        next_state = np.random.choice(range(self.n_states), p=next_state_probs)

        # Set next state
        self.state = next_state     

        return next_state, (next_state == 31 or (self.steps >= self.max_steps))

    def transition(self, state, action, next_state=None):
        # Rows and Columns
        n_rows = 7
        n_cols = 5

        # Initialize next states array
        next_states = []

        # Top Left Corner
        if state == 0:
            if action == 1:
                next_states.append(state + 1)
            elif action == 2:
                next_states.append(state + n_cols)
            else:
                next_states.append(state)
        
        # River
        elif state in [1, 6, 11, 16, 21, 26]:
            next_states.append(state + n_cols)
        elif state == 31:
            next_states.append(state)
        
        # Left Wall
        elif state in [10, 15, 20]:
            if action == 0:
                next_states.append(state - n_cols)
            elif action == 1:
                next_states.append(state + 1)
            elif action == 2:
                next_states.append(state + n_cols)
            else:
                next_states.append(state)

        # Bottom Left Corner
        elif state == 30:
            if action == 1:
                next_states.append(state + 1)
            elif action == 0:
                next_states.append(state - n_cols)
            else:
                next_states.append(state)

        # Bridge
        elif state in [5, 25]:
            if action == 0:
                next_states.append(state - n_cols)
            elif action == 1:
                next_states.append(state + 2)
            elif action == 2:
                next_states.append(state + n_cols)
            else:
                next_states.append(state)

        # Noise
        elif state in [2, 3, 4, 7, 8, 9, 12, 13, 14]:
            next_states.extend([2, 3, 4, 7, 8, 9, 12, 13, 14])
        
        # Grid
        elif state in [22, 23]:
            if action == 1:
                next_states.append(state + 1)
            elif action == 2:
                next_states.append(state + n_cols)
            elif action == 3:
                next_states.append(state - 1)
            else:
                next_states.append(state)
        
        elif state == 24:
            if action == 2:
                next_states.append(state + n_cols)
            elif action == 3:
                next_states.append(state - 1)
            else:
                next_states.append(state)

        elif state in [27, 28]:
            if action == 0:
                next_states.append(state - n_cols)
            elif action == 1:
                next_states.append(state + 1)
            elif action == 2:
                next_states.append(state + n_cols)
            elif action == 3:
                next_states.append(state - 1)
            else:
                next_states.append(state)
        
        elif state == 29:
            if action == 0:
                next_states.append(state - n_cols)
            elif action == 2:
                next_states.append(state + n_cols)
            elif action == 3:
                next_states.append(state - 1)
            else:
                next_states.append(state)

        elif state in [32, 33]:
            if action == 0:
                next_states.append(state - n_cols)
            elif action == 1:
                next_states.append(state + 1)
            elif action == 3:
                next_states.append(state - 1)
            else:
                next_states.append(state)

        elif state == 34:
            if action == 0:
                next_states.append(state - n_cols)
            elif action == 3:
                next_states.append(state - 1)
            else:
                next_states.append(state)
        
        # Initialize Transition array
        transition_array = np.zeros(self.n_states)

        # Set all next states to 1
        transition_array[next_states] = 1

        # Normalize transition array 
        if sum(transition_array) > 0:
            transition_array /= sum(transition_array)

        # Return the transition probabilities
        if next_state == None:
            return transition_array
        else:
            return transition_array[next_state]

    def simulate_and_plot_trajectories(self, policy, n_trajectories=5):
        
        # Grid shape 
        grid_shape = (7, 5)
        rows, cols = grid_shape
        
        # Define specific background colors for zones
        background_idx = np.zeros(grid_shape)
        
        river_states = [1, 6, 11, 16, 21, 26, 31]
        wall_states = [17, 18, 19]
        noise_states = [2, 3, 4, 7, 8, 9, 12, 13, 14]
        bridge_states = [5, 25]
        empowerment_states = [22, 23, 24, 27, 28, 29, 32, 33, 34] 

        for s in range(self.n_states):
            r, c = s // cols, s % cols
            if s in river_states:
                background_idx[r, c] = 1
            elif s in wall_states:
                background_idx[r, c] = 2
            elif s in noise_states:
                background_idx[r, c] = 3
            elif s in empowerment_states:
                background_idx[r, c] = 4
            elif s in bridge_states:
                background_idx[r, c] = 5

        # Custom colormap reflecting specified requirements
        cmap = ListedColormap(['white', 'darkgray', 'black', 'royalblue', 'crimson', 'saddlebrown'])

        fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))
        ax.imshow(background_idx, cmap=cmap, origin="upper", extent=[-0.5, cols-0.5, rows-0.5, -0.5])

        # Simulate trajectories
        for t in range(n_trajectories):
            state = self.reset()
            states_visited = [state]
            done = False
            
            # Initialize history variables
            past_state = -1   
            past_action = -1
            
            while not done:
                prob_distribution = policy[past_state, past_action, state] 
                action = np.random.choice(range(self.n_actions), p=prob_distribution)
                next_state, done = self.step(action)
                states_visited.append(next_state)
                
                # Step history forward before updating current state
                past_state = state
                past_action = action
                state = next_state

            # Coordinate generation for paths
            x_coords = [s % cols for s in states_visited]
            y_coords = [s // cols for s in states_visited]
            
            # Slight jitter added so overlapping trajectory tracks remain visible
            jitter_x = np.random.uniform(-0.08, 0.08, size=len(x_coords))
            jitter_y = np.random.uniform(-0.08, 0.08, size=len(y_coords))
            
            ax.plot(np.array(x_coords) + jitter_x, np.array(y_coords) + jitter_y, 
                    marker='o', markersize=5, linewidth=2, label=f'Run {t+1}')

        # Text labels mapping
        for s in range(self.n_states):
            r, c = s // cols, s % cols
            text_color = "white" if s in wall_states or s in empowerment_states else "black"
            ax.text(c, r, f"S_{s}", ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')

        # Grid adjustments
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="gold", linestyle="-", linewidth=1)
        
        # Dynamic Legend
        grey_patch = mpatches.Patch(color='darkgray', label='River')
        black_patch = mpatches.Patch(color='black', label='Walls')
        blue_patch = mpatches.Patch(color='royalblue', label='Noise Zone')
        red_patch = mpatches.Patch(color='crimson', label='Empowerment')
        brown_patch = mpatches.Patch(color='saddlebrown', label='Bridges')

        plt.legend(handles=[grey_patch, black_patch, blue_patch, red_patch, brown_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"Simulation of {n_trajectories} Environment Trajectories", fontsize=12, pad=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('simulations.pdf', dpi=300, bbox_inches='tight')
