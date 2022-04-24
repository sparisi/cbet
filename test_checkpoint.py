import numpy as np
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tqdm import tqdm

import src.models as models
from src.env_utils import make_gym_env, make_environment

import argparse

PolicyNet = models.PolicyNet

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--n_episodes', type=int, default=100)
parser.add_argument('--n_seeds', type=int, default=10)
parser.add_argument('--to_env', default='MiniGrid-Unlock-v0,' + \
                                        'MiniGrid-DoorKey-8x8-v0,' + \
                                        'MiniGrid-KeyCorridorS3R3-v0,' + \
                                        'MiniGrid-UnlockPickup-v0,' + \
                                        'MiniGrid-BlockedUnlockPickup-v0,' + \
                                        'MiniGrid-MultiRoom-N6-v0,' + \
                                        'MiniGrid-MultiRoom-N12-S10-v0,' + \
                                        'MiniGrid-ObstructedMaze-1Dlh-v0,' + \
                                        'MiniGrid-ObstructedMaze-2Dlh-v0,' + \
                                        'MiniGrid-ObstructedMaze-2Dlhb-v0'
)



def test_model(model, keys, flags):
    torch.manual_seed(flags.seed)
    torch.cuda.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    env = make_environment(flags)

    env_output = env.initial()
    agent_state = model.initial_state(batch_size=1)
    agent_output, unused_state = model(env_output, agent_state)

    stats = dict()
    for key in keys:
        stats.update({key: []})
    stats.update({'action': []})

    for episode in tqdm(range(flags.n_episodes)):
        if 'interactions' in keys:
            inters = [] # Unique interactions per episode
        # 1000 max steps because Habitat and MiniGrid store it in different variables, and 1000 is enough for both
        for step in tqdm(range(1000), leave=False, disable=not flags.verbose):
            with torch.no_grad():
                agent_output, agent_state = model(env_output, agent_state)
            env_output = env.step(agent_output['action'])
            stats['action'].append(agent_output['action'].item())
            assert float(env_output['interactions'].numpy()) <= 1, 'error in inter'
            if 'interactions' in keys:
                inters.append(float(env_output['interactions'].numpy()))
            if env_output['done']:
                break

        if flags.verbose:
            print(flush=True)

        for key in keys:
            if key == 'interactions':
                stats[key].append(np.array(inters).sum())
            else:
                stats[key].append(float(env_output[key].numpy()[0][0]))
            if flags.verbose:
                print(key, env_output[key].numpy()[0][0], '  ', end='')

        if flags.verbose:
            print(flush=True)

    print(' ', flags.seed, end='')
    for key in keys:
        print((' - %s: %f') % (key, np.mean(stats[key])), end='')

    print()
    print(flush=True)

    env.close()

    return stats


def run(flags):
    flags.device = None
    flags.fixed_seed = None

    if torch.cuda.is_available():
        print('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        print('Not using CUDA.')
        flags.device = torch.device('cpu')

    keys = ['episode_return', 'episode_step', 'episode_win', 'interactions', 'visited_states']
    envs = flags.to_env.split(',')

    stats = dict()

    tmp_env = make_gym_env(envs[0])
    model = PolicyNet(tmp_env.observation_space.shape, tmp_env.action_space.n, envs[0])
    tmp_env.close()

    for env_id in envs:
        flags.env = env_id
        if 'MiniGrid' in env_id:
            flags.no_reward = False
        else:
            flags.no_reward = True
        print(' ', env_id)

        for seed in range(1, flags.n_seeds + 1):
            flags.seed = seed
            flags.run_id = seed
            flags.xpid = ''
            flags.savedir = ''

            if flags.checkpoint: # do not pass a checkpoint to test random policy
                checkpoint = torch.load(flags.checkpoint)
                model.load_state_dict(checkpoint["actor_model_state_dict"])

            model.share_memory()
            stats.update({(env_id, seed): (test_model(model, keys, flags))})



if __name__ == '__main__':
    flags = parser.parse_args()
    run(flags)
