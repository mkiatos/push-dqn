import numpy as np


def run_episode(env, agent, mdp, max_steps=50, train=False, seed=0):
    """
    Runs an episode. Useful for training or evaluating RL agents.

    Parameters
    ----------
    env : Env
        An object of an environment.
    agent : Agent
        An object of an Agent.
    agent : MDP
        An object of an MDP.
    max_steps : int
        The maximum steps to run the episode.
    train : bool
        Set True for a training episode and False for an evaluation episode. In particular, if true the agent.explore()
        will be called instead of agent.predict(). Also the agent.learn() will be called for updating the agent's
        policy.

    Returns
    -------
    list :
        A list of N dictionaries, where N the total number of timesteps the episode performed. Each dictionary contains
        data for each step, such as the Q-value, the reward and the terminal state's id.
    """

    episode_data = []

    env.seed(seed)
    mdp.seed(seed)
    obs = env.reset()

    # Keep reseting the environment if the initial state is not valid according to the MDP
    while not mdp.init_state_is_valid(obs):
        obs = env.reset()

    for i in range(max_steps):
        print('-- Step :', i)

        # Transform observation from env (e.g. RGBD, mask) to state representation from MDP (e.g. the latent from an
        #   autoencoder)
        state = mdp.state_representation(obs)

        # Select action
        if train:
            action = agent.explore(state)
        else:
            action = agent.predict(state)

        print('action:', action)

        # Transform an action from the agent (e.g. -1, 1) to an env action: (e.g. 2 3D points for a push)
        env_action = mdp.action(obs, action)

        # Step environment
        next_obs = env.step(env_action)

        # Calculate reward from environment state
        reward = mdp.reward(obs, next_obs, action)
        print('reward:', reward)

        # Calculate terminal state
        terminal_id = mdp.terminal(obs, next_obs)

        # Log
        if terminal_id == 1:
            raise RuntimeError('Terminal id = 1 is taken for maximum steps.')

        if -10 < terminal_id <= 0 and i == max_steps - 1:
            terminal_id = 1  # Terminal state 1 means terminal due to maximum steps reached

        timestep_data = {"q_value": 0,
                         "reward": reward,
                         "terminal_class": terminal_id,
                         "action": action,
                         "obs": copy.deepcopy([x.dict() for x in obs['full_state']['objects']]),
                         "agent": copy.deepcopy(agent.info)
                         }
        episode_data.append(timestep_data)

        print('terminal state', terminal_id)

        # If the mask is empty, stop the episode
        if terminal_id <= -10:
            break

        if train:
            next_state = mdp.state_representation(next_obs)
            # Agent should use terminal as true/false
            transition = Transition(state, action, reward, next_state, bool(terminal_id))
            agent.learn(transition)

        obs = copy.deepcopy(next_obs)

        if terminal_id > 0:
            break

        print('-----------------')

    return episode_data


if __name__ == '__main__':
    pass
	

