import numpy as np
from tqdm import tqdm

def entropy(x, axis=-1):
    """Compute entropy along a specific axis safely."""
    x = np.asarray(x)
    
    # Mask out zeros to avoid log2 warnings and NaNs
    mask = x > 0
    ent = np.zeros_like(x)
    ent[mask] = -x[mask] * np.log2(x[mask])
    return np.sum(ent, axis=axis)

def plasticity_with_memory(policy_mem, initial_distribution, transition_dynamics, n_states, n_actions, steps=1000):
    """
    Calculate the Plasticity of an agent in an environment given the augmented policy with memory.
    
    This function tracks the Markov chain transitions over an augmented state space S_aug = (s_t, s_{t-1}, a_{t-1}) to calculate:
    I(A; S_aug) = H(A) - H(A | S_aug)
    """
    # Add +1 to state/action dimensions to accommodate boundary conditions (t=0, where history is empty / -1)
    n_states_hist = n_states + 1
    n_actions_hist = n_actions + 1

    # Total size of the augmented state space: S_t * S_{t-1} * A_{t-1}
    n_aug = n_states * n_states_hist * n_actions_hist
    
    # Evaluate and store base transition dynamics P(s_{t+1} | s_t, a_t) in a 3D NumPy array
    T_base = np.zeros((n_states, n_actions, n_states))
    for sc in range(n_states):
        for ac in range(n_actions):
            for sn in range(n_states):
                T_base[sc, ac, sn] = transition_dynamics(sc, ac, sn)
                
    # Helper to map the augmented state tuple (s_t, s_{t-1}, a_{t-1}) to a unique flat 1D index
    def get_idx(sc, sp, ap):
        if sp == -1: sp = n_states      
        if ap == -1: ap = n_actions  
        return sc * (n_states_hist * n_actions_hist) + sp * n_actions_hist + ap

    # Transition matrix representing transitions between successive augmented states: P(S_aug' | S_aug)
    aug_transition = np.zeros((n_aug, n_aug))
    for sc in range(n_states):
        for sp in range(n_states_hist):
            for ap in range(n_actions_hist):
                idx_curr = get_idx(sc, sp, ap)
                for ac in range(n_actions):
                    
                    # Fetch policy probability: P(a_t | s_{t-1}, a_{t-1}, s_t)
                    prob_a = policy_mem[sp, ap, sc, ac]
                    if prob_a > 0:
                        for sn in range(n_states):
                            
                            # Fetch transition probability: P(s_{t+1} | s_t, a_t)
                            prob_s = T_base[sc, ac, sn]
                            if prob_s > 0:
                                
                                # The next augmented state updates to (s_{t+1}, s_t, a_t)
                                idx_next = get_idx(sn, sc, ac)
                                
                                # Combined transition probability is the product of policy choice and dynamics
                                aug_transition[idx_curr, idx_next] += prob_a * prob_s

    # Initialize the probability distribution over augmented states at t=0
    p_sa_aug = np.zeros(n_aug)
    for s0, prob_s0 in enumerate(initial_distribution):
        if prob_s0 > 0:
            p_sa_aug[get_idx(s0, -1, -1)] = prob_s0

    # Flatten the policy actions mapping directly to their flat augmented indices
    flat_policy = np.zeros((n_aug, n_actions))
    for sc in range(n_states):
        for sp in range(n_states_hist):
            for ap in range(n_actions_hist):
                idx = get_idx(sc, sp, ap)
                flat_policy[idx, :] = policy_mem[sp, ap, sc, :]

    # Precalculate action selection entropy for every individual augmented state: H(A | S_aug)
    H_AS_array = entropy(flat_policy, axis=1)

    step_plasticity = np.zeros(steps)

    # Simulation Loop
    for t in range(steps):

        # Marginal action probability: P(a_t) = sum_{S_aug} P(S_aug) * P(a_t | S_aug)
        p_a_curr = p_sa_aug @ flat_policy
        
        # Calculate global action entropy: H(A)
        H_A = entropy(p_a_curr)
        
        # Calculate expected conditional action entropy: E_{S_aug}[H(A | S_aug)]
        H_AS = np.dot(H_AS_array, p_sa_aug)
        
        # Plasticity at step t: Mutual Information I(A; S_aug) = H(A) - H(A | S_aug)
        step_plasticity[t] = H_A - H_AS
        
        # Propagate the augmented state probability distribution forward by one step using Markov chain multiplication
        p_sa_aug = p_sa_aug @ aug_transition
        
    # Compute the cumulative moving average over steps to smooth transient dynamics and evaluate convergence
    plasticity_history = np.cumsum(step_plasticity) / np.arange(1, steps + 1)
    
    # Return the final converged plasticity value and the complete running history
    return plasticity_history[-1], plasticity_history.tolist()

def empowerment_with_memory(policy_mem, initial_distribution, transition_dynamics, n_states, n_actions, steps=1000):
    """
    Calculate the Empowerment of an agent with memory in an environment.
    
    This function tracks the Markov chain transitions over an augmented state space 
    S_aug = (s_t, s_{t-1}, a_{t-1}) to calculate the mutual information:
    I(S'; A | S_aug) = H(S' | S_aug) - H(S' | S_aug, A)
    """

    # Add +1 to state/action dimensions to accommodate boundary conditions
    n_states_hist = n_states + 1
    n_actions_hist = n_actions + 1
    
    # Total size of the augmented state space: S_t * S_{t-1} * A_{t-1}
    n_aug = n_states * n_states_hist * n_actions_hist

    # Evaluate and store base transition dynamics P(s_{t+1} | s_t, a_t) in a 3D NumPy array
    T_base = np.zeros((n_states, n_actions, n_states))
    for sc in range(n_states):
        for ac in range(n_actions):
            for sn in range(n_states):
                T_base[sc, ac, sn] = transition_dynamics(sc, ac, sn)
                
    # Calculate environmental uncertainty: H(S' | S, A) for all state-action pairs
    H_env_noise = entropy(T_base, axis=2)

    # Helper to map the augmented state tuple (s_t, s_{t-1}, a_{t-1}) to a unique flat 1D index
    def get_idx(sc, sp, ap):
        if sp == -1: sp = n_states      
        if ap == -1: ap = n_actions
        return sc * (n_states_hist * n_actions_hist) + sp * n_actions_hist + ap

    # Transition matrix representing transitions between successive augmented states: P(S_aug' | S_aug)
    aug_transition = np.zeros((n_aug, n_aug))
    for sc in range(n_states):
        for sp in range(n_states_hist):
            for ap in range(n_actions_hist):
                idx_curr = get_idx(sc, sp, ap)
                for ac in range(n_actions):
                    
                    # Fetch policy probability: P(a_t | s_{t-1}, a_{t-1}, s_t)
                    prob_a = policy_mem[sp, ap, sc, ac]
                    if prob_a > 0:
                        for sn in range(n_states):
                            
                            # Fetch transition probability: P(s_{t+1} | s_t, a_t)
                            prob_s = T_base[sc, ac, sn]
                            if prob_s > 0:
                                
                                # The next augmented state updates to (s_{t+1}, s_t, a_t)
                                idx_next = get_idx(sn, sc, ac)
                                
                                # Combined transition probability is the product of policy choice and dynamics
                                aug_transition[idx_curr, idx_next] += prob_a * prob_s

    # Initialize the probability distribution over augmented states at t=0
    p_aug = np.zeros(n_aug)
    for s0, prob_s0 in enumerate(initial_distribution):
        if prob_s0 > 0:
            
            # At step 0, previous state and previous action are set to the -1 (empty) placeholder
            p_aug[get_idx(s0, -1, -1)] = prob_s0

    # Arrays initialized to bypass nested loop lookups during time-step simulation
    flat_policy = np.zeros((n_aug, n_actions))
    env_noise_mapped = np.zeros((n_aug, n_actions))
    p_sn_given_aug = np.zeros((n_aug, n_states))
    
    for sc in range(n_states):
        for sp in range(n_states_hist):
            for ap in range(n_actions_hist):
                idx = get_idx(sc, sp, ap)
                
                # Flatten the policy actions mapping directly to their flat augmented indices
                flat_policy[idx, :] = policy_mem[sp, ap, sc, :]
                
                # Map environment transition noise directly to the corresponding augmented indices
                env_noise_mapped[idx, :] = H_env_noise[sc, :]
                
                # Calculate next base-state distribution given augmented state: P(s_{t+1} | S_aug) = sum_a P(a|S_aug)*P(s_{t+1}|s,a)
                p_sn_given_aug[idx, :] = flat_policy[idx, :] @ T_base[sc]

    # Precalculate entropy of the next state given augmented state: H(S' | S_aug)
    H_s_given_aug_array = entropy(p_sn_given_aug, axis=1) 
    
    # Precalculate expected environmental noise: H(S' | S_aug, A) = sum_a P(a|S_aug) * H(S'|s, a)
    H_s_given_aug_act_array = np.sum(flat_policy * env_noise_mapped, axis=1)

    step_empowerment = np.zeros(steps)

    # Simulation Loop
    for t in range(steps):
        
        # Expected entropy of next state over the current augmented state distribution: E_{S_aug}[H(S' | S_aug)]
        h_s_given_history = np.dot(p_aug, H_s_given_aug_array)
        
        # Expected environmental noise over the current augmented state distribution: E_{S_aug}[H(S' | S_aug, A)]
        h_s_given_history_action = np.dot(p_aug, H_s_given_aug_act_array)
        
        # Mutual information at step t: I(S'; A | S_aug) = H(S'|S_aug) - H(S'|S_aug, A)
        step_empowerment[t] = h_s_given_history - h_s_given_history_action
        
        # Propagate the augmented state probability distribution forward by one step using Markov chain multiplication
        p_aug = p_aug @ aug_transition

    # Compute the cumulative moving average over steps to smooth transient dynamics and evaluate convergence
    empowerment_history = np.cumsum(step_empowerment) / np.arange(1, steps + 1)
    
    # Return the final converged empowerment value and the complete running history
    return empowerment_history[-1], empowerment_history.tolist()