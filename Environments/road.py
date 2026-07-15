import numpy as np

class Road:
    def __init__(self, maxsteps=10):

        # States: {-2, -1, 0, 1, 2} 
        self.n_states = 5 

        # Actions: {-1, 0, 1}
        self.n_actions = 3

        # Max steps
        self.maxsteps = maxsteps

        # Initial State Distribution
        self.initial_state_distribution = np.array([1/5, 1/5, 1/5, 1/5, 1/5])

        # Initial State Distribution Function
        self.init_distribution_function = lambda s: self.initial_state_distribution[s]

        # Game Parameters
        self.state = np.random.randint(self.n_states)
        self.steps = 0

        # Transition Matrix
        self.transition_matrix = np.array([
            [
                [1, 0, 0, 0, 0],
                [2/3, 1/3, 0, 0, 0],
                [1/3, 1/3, 1/3, 0, 0],
            ], 
            [
                [2/3, 1/3, 0, 0, 0],
                [1/3, 1/3, 1/3, 0, 0],
                [0, 1/3, 1/3, 1/3, 0],
            ], 
            [
                [1/3, 1/3, 1/3, 0, 0],
                [0, 1/3, 1/3, 1/3, 0],
                [0, 0, 1/3, 1/3, 1/3],
            ],
            [
                [0, 1/3, 1/3, 1/3, 0],
                [0, 0, 1/3, 1/3, 1/3],
                [0, 0, 0, 1/3, 2/3],
            ], 
            [
                [0, 0, 1/3, 1/3, 1/3],
                [0, 0, 0, 1/3, 2/3],
                [0, 0, 0, 0, 1]
            ],
        ])

    def reset(self):
        # Reset steps and state
        self.state = np.random.randint(self.n_states)
        self.steps = 0
        return self.state
    
    def step(self, action):
        # Increment steps
        self.steps += 1

        # Get next state probabilities
        next_state_probs = self.transition_function(self.state, action)

        # Sample next state
        next_state = np.random.choice(range(self.n_states), p=next_state_probs)

        # Set next state
        self.state = next_state

        return next_state, (self.steps >= self.maxsteps)
    
    def transition_function(self, s, a, s1=None):

        # Get transitions from transition matrix
        if s1 is not None:

            # Return transition probability
            return self.transition_matrix[s][a][s1]
        else:

            # Return transition probability array if next state is None
            return self.transition_matrix[s][a]