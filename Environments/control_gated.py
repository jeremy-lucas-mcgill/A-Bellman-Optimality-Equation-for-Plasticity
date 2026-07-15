import numpy as np 

class ControlGatedMDP:
    def __init__(self, seed=None, maxsteps=10):
        # States and Actions
        self.n_states = 2
        self.n_actions = 2

        # Max steps
        self.max_steps = maxsteps

        # Initial State Distribution
        self.initial_state_distribution = np.array([0.5,0.5])

        # Transition Matrix
        self.matrix = np.array([
            [[0.5, 0.5], [0.5, 0.5]],
            [[1.0, 0.0], [0.0, 1.0]],
        ])

        # Transition Function
        self.transition_function = lambda s, a, s1=None: self.channel_transition(s,a, s1)
        
        # Initial State Distribution Function
        self.init_distribution_function = lambda s: self.initial_state_distribution[s]

    
    def reset(self):
        # Reset steps and state
        self.steps = 0
        self.state = np.random.choice(range(self.n_states))
        return self.state
    
    def step(self, action):
        # Increment state
        self.steps += 1
        
        # Get next probabilities
        next_state_probs = self.transition_function(self.state, action)

        # Sample the next state 
        next_state = np.random.choice(range(self.n_states), p=next_state_probs)

        # Set next state
        self.state = next_state

        return next_state, (self.steps >= self.max_steps)

    def channel_transition(self, state, action, next_state=None):
        
        # Get transitions from transition matrix
        if next_state is not None:
            
            # Return transition probability
            return self.matrix[state][action][next_state]
        else:
            
            # Return transition probability array if next state is None
            return self.matrix[state][action]

    def generate_random_transition_matrix(self, seed=None):
        # Create a local generator
        rng = np.random.default_rng(seed)
        
        # Use the generator to sample from the Dirichlet distribution
        matrix = rng.dirichlet([1.0] * self.n_states, size=(self.n_states, self.n_actions))
        
        return matrix
