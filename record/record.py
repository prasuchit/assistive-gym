import torch as th
import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
from ray.rllib.agents import ppo, sac
from ray.tune.logger import pretty_print
from numpngw import write_apng
from tqdm import tqdm
# import shutup; shutup.please()


def setup_config(env, algo, coop=False, seed=0, extra_configs={}):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [100, 100]
    elif algo == 'sac':
        # NOTE: pip3 install tensorflow_probability
        config = sac.DEFAULT_CONFIG.copy()
        config['timesteps_per_iteration'] = 400
        config['learning_starts'] = 1000
        config['Q_model']['fcnet_hiddens'] = [100, 100]
        config['policy_model']['fcnet_hiddens'] = [100, 100]
        # config['normalize_actions'] = False
    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    # if algo == 'sac':
    #     config['num_workers'] = 1
    if coop:
        obs = env.reset()
        policies = {'robot': (None, env.observation_space_robot, env.action_space_robot, {}), 'human': (None, env.observation_space_human, env.action_space_human, {})}
        config['multiagent'] = {'policies': policies, 'policy_mapping_fn': lambda a: a}
        config['env_config'] = {'num_agents': 2}
    return {**config, **extra_configs}

def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'sac':
        agent = sac.SACTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                agent.restore(checkpoint_path)
                # return agent, checkpoint_path
            return agent, None
    return agent, None

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

def record_policy(env, env_name, algo, policy_path, coop=True, seed=0, num_steps=200, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, coop, seed=seed)
        
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)
    
    env.render()
    obs = env.reset()

    if coop:
        states_rollout = {'robot': [], 'human': []}
        next_states_rollout = {'robot': [], 'human': []}
        actions_rollout = {'robot': [], 'human': []}
        rewards_rollout = {'robot': [], 'human': []}
        dones_rollout = {'robot': [], 'human': []}
        infos_rollout = {'robot': [], 'human': []}
    else:
        raise NotImplementedError

    length_stats = []
    reward_stats = []

    length = 0
    reward = 0

    for step in tqdm(range(num_steps)):
        if coop:
            action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
            action_human = test_agent.compute_action(obs['human'], policy_id='human')
            next_obs, rewards, done, info = env.step({'robot': action_robot, 'human': action_human})

            states_rollout['robot'].append(obs['robot'])
            states_rollout['human'].append(obs['human'])
            next_states_rollout['robot'].append(next_obs['robot'])
            next_states_rollout['human'].append(next_obs['human'])
            actions_rollout['robot'].append(action_robot)
            actions_rollout['human'].append(action_human)
            rewards_rollout['robot'].append(rewards['robot'])
            rewards_rollout['human'].append(rewards['human'])
            dones_rollout['robot'].append(done['robot'])
            dones_rollout['human'].append(done['human'])
            infos_rollout['robot'].append(info['robot'])
            infos_rollout['human'].append(info['human'])

            done = done['__all__']
            length += 1
            reward += (rewards['robot'] + rewards['human']) / 2

            if done:
                obs = env.reset()

                length_stats.append(length)
                reward_stats.append(reward)

                length = 0
                reward = 0
            else:
                obs = next_obs

    states_rollout['robot'] = th.as_tensor(states_rollout['robot']).float()
    states_rollout['human'] = th.as_tensor(states_rollout['human']).float()
    actions_rollout['robot'] = th.as_tensor(actions_rollout['robot']).float()
    actions_rollout['human'] = th.as_tensor(actions_rollout['human']).float()
    rewards_rollout['robot'] = th.as_tensor(rewards_rollout['robot']).float()
    rewards_rollout['human'] = th.as_tensor(rewards_rollout['human']).float()
    next_states_rollout['robot'] = th.as_tensor(next_states_rollout['robot']).float()
    next_states_rollout['human'] = th.as_tensor(next_states_rollout['human']).float()
    dones_rollout['robot'] = th.as_tensor(dones_rollout['robot']).float()
    dones_rollout['human'] = th.as_tensor(dones_rollout['human']).float()
    
    trajectories = {
        'state': states_rollout,
        'action': actions_rollout,
        'reward': rewards_rollout,
        'done': dones_rollout,
        'next_state': next_states_rollout,
        'info': infos_rollout
    }

    save_path = f'./trajectory/{env_name}'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    th.save(trajectories, f'{save_path}/trajectory.pt')

    print(f'Collect Episodes: {len(length_stats)} | Avg Length: {round(np.mean(length_stats), 2)} | Avg Reward: {round(np.mean(reward_stats), 2)} | Std Reward: {round(np.std(reward_stats), 2)}')    

    env.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='FeedingSawyerHuman-v1',
                        help='Environment to train on (default: FeedingSawyerHuman-v1)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    coop = ('Human' in args.env)
    checkpoint_path = None

    record_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, seed=args.seed)
