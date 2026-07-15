import numpy as np
from metrics import softmax
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import itertools

import numpy as np
import itertools

class Agent:
    def __init__(self, n_states, n_actions, gamma, transition_function, initial_state_distribution):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.transition_function = transition_function
        self.initial_state_distribution = initial_state_distribution 
        
        # Initialize Policy Logits
        self.policy_logits = np.zeros(
            (self.n_states + 1, self.n_actions + 1, self.n_states, self.n_actions)
        )

        # Initialize Value Table
        self.V_table = np.zeros((self.n_states + 1, self.n_actions + 1))

        # Precompute a static transition tensor
        self.T = np.zeros((self.n_states, self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for s1 in range(self.n_states):
                    self.T[s, s1, a] = self.transition_function(s, a, s1)

    def get_policy(self):
        """
        Returns a deterministic one-hot argmax policy.
        """
        # Find the index of the max value along the final action axis
        best_actions = np.argmax(self.policy_logits, axis=-1)
        
        # Construct a one-hot tensor matching the target shape
        policy = np.zeros_like(self.policy_logits)
        
        # Advanced indexing to set the argmax actions to 1.0
        idx_sh, idx_ah, idx_s = np.indices(best_actions.shape)
        policy[idx_sh, idx_ah, idx_s, best_actions] = 1.0
        return policy

    def update(self, precision=2):
        """
        Updates the value table and policy logits using the knapsack dynamic 
        programming solver.
        """
        new_V_table = np.copy(self.V_table)
        new_policy_logits = np.copy(self.policy_logits)

        # Generate all history pairs
        history_pairs = [(sh, ah) for sh in range(self.n_states) for ah in range(self.n_actions)]
        history_pairs.append((-1, -1))

        for sh, ah in history_pairs:
            # Determine state probabilities for this history
            if sh == -1 and ah == -1:
                state_probs = np.array([self.initial_state_distribution(s) for s in range(self.n_states)])
            else:
                state_probs = np.array([self.transition_function(sh, ah, s) for s in range(self.n_states)])

            state_probs = np.round(state_probs, precision)
                
            if np.all(state_probs <= 1e-12):
                continue

            # Extract values for future states across all available actions
            value_matrix = self.V_table[:self.n_states, :self.n_actions]

            # Solve the optimization via DP
            best_score, best_combination = self.solve_knapsack_dp(
                transition_weights=state_probs, 
                value_matrix=value_matrix, 
                gamma=self.gamma
            )

            # Reconstruct the one-hot deterministic policy slice from optimal choices
            best_policy_slice = np.zeros((self.n_states, self.n_actions))
            for s, chosen_action in enumerate(best_combination):
                best_policy_slice[s, chosen_action] = 1.0

            # Write directly to our update tables using the specific history indices
            new_V_table[sh, ah] = best_score
            new_policy_logits[sh, ah] = best_policy_slice

        # Update Value Table and Policy Logits
        self.V_table = new_V_table
        self.policy_logits = new_policy_logits

    def solve_knapsack_dp(self, transition_weights, value_matrix):
        """
        Optimizes the multi-weight knapsack decision paths using Dynamic Programming.
        
        Parameters:
        -----------
        transition_weights : list or np.ndarray
            Array of length self.n_states containing transition probabilities/weights.
        value_matrix : list of lists or np.ndarray
            Matrix of shape (self.n_states, self.n_actions) containing future state values.
            
        Returns:
        --------
        best_score : float
            The absolute maximum score achievable.
        best_combination : tuple
            The optimal assignment of actions chosen for each state.
        """

        # Enforce shape alignment with class parameters
        assert len(transition_weights) == self.n_states, f"Expected {self.n_states} transition weights."
        assert len(value_matrix) == self.n_states and len(value_matrix[0]) == self.n_actions, \
            f"Value matrix must strictly be of shape ({self.n_states}, {self.n_actions})"

        # DP State initialization: tracked actions length is always (n_actions - 1)
        dp_states = {tuple(np.zeros(self.n_actions - 1)): (0.0, ())}

        # DP Search Loop
        for i, t in enumerate(transition_weights):
            next_dp_states = {}
            
            # Expand currently reached active states
            for current_state, (current_val, path) in dp_states.items():
                
                # Branch out choices for all available actions
                for o in range(self.n_actions):
                    new_current_state = list(current_state)
                    
                    if o < self.n_actions - 1:
                        new_current_state[o] += t 
                    
                    new_state_tuple = tuple(new_current_state)

                    # Linear value step calculation
                    linear_gain = t * (self.gamma * value_matrix[i][o])
                    new_val = current_val + linear_gain
                    new_path = path + ((i, o),)

                    # Overwrite prune check
                    if new_state_tuple in next_dp_states:
                        if new_val > next_dp_states[new_state_tuple][0]:
                            next_dp_states[new_state_tuple] = (new_val, new_path)
                    else:
                        next_dp_states[new_state_tuple] = (new_val, new_path)
                    
            dp_states = next_dp_states

        # Evaluating DP Final Layer
        best_score = -np.inf
        best_path = None
        sum_transition_weights = sum(transition_weights)

        for state, (linear_val, path) in dp_states.items():
            q_sum = np.zeros(self.n_actions)
            for o in range(self.n_actions - 1):
                q_sum[o] = state[o]
            
            # Remainder conservation logic for the final action
            q_sum[self.n_actions - 1] = sum_transition_weights - sum(q_sum[:-1])
            
            # Vectorized safety stability offset
            q_sum += 1e-12
            log_sum_penalty = sum(q_sum[o] * (-np.log(q_sum[o])) for o in range(self.n_actions))
            
            final_score = linear_val + log_sum_penalty
            
            if final_score > best_score:
                best_score = final_score
                best_path = path

        # Parse recorded paths back into sequential format
        best_combination = [None] * self.n_states
        for i, o in best_path:
            best_combination[i] = o

        return best_score, tuple(best_combination)