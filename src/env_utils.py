# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import gym
import torch
from collections import deque, defaultdict
from gym import spaces
import numpy as np
import itertools
from copy import deepcopy
import os
import pathlib

from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

try:
    import quaternion
    if 'VERBOSE_HABITAT' not in os.environ: # To suppress Habitat messages
        os.environ['MAGNUM_LOG'] = 'quiet'
        os.environ['GLOG_minloglevel'] = '2'
        os.environ['HABITAT_SIM_LOG'] = 'quiet'
    else:
        os.environ['GLOG_minloglevel'] = '0'
        os.environ['MAGNUM_LOG'] = 'verbose'
        os.environ['MAGNUM_GPU_VALIDATION'] = 'ON'

    import habitat
    from habitat_baselines.config.default import get_config
    from habitat_baselines.common.environments import get_env_class
    from habitat_baselines.utils.env_utils import make_env_fn
    from habitat.utils.visualizations.utils import observations_to_image
    from habitat.datasets.pointnav.pointnav_generator import is_compatible_episode
except:
    print('WARNING! Could not import Habitat.')

#from src.viewer import ImageViewer # Must be imported AFTER habitat


# ------------------------------------------------------------------------------


gym.envs.register(
    id='MiniGrid-MultiRoom-N12-S10-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnv',
    kwargs={'minNumRooms' : 12, \
            'maxNumRooms' : 12, \
            'maxRoomSize' : 10},
)

gym.envs.register(
    id='MiniGrid-MultiRoom-N7-S4-v0',
    entry_point='gym_minigrid.envs:MultiRoomEnv',
    kwargs={'minNumRooms' : 7, \
            'maxNumRooms' : 7, \
            'maxRoomSize' : 4},
)

gym.envs.register(
    id='MiniGrid-MultiRoomNoisyTV-N7-S4-v0',
    entry_point='src:MultiRoomNoisyTVEnv',
    kwargs={'minNumRooms' : 7, \
            'maxNumRooms' : 7, \
            'maxRoomSize' : 4},
)

# ------------------------------------------------------------------------------


def _format_observation(obs):
    obs = torch.squeeze(torch.tensor(obs))
    return obs.view((1, 1) + obs.shape)


# ------------------------------------------------------------------------------


# Habitat goal location is always the same, since no information about it is in
# the observation (e.g., no marker on goals).
# Start location depends on the run seed, and is kept fixed during all episode
# of the same run. This makes exploration harder.
def _sample_start_and_goal(sim, seed, number_retries_per_target=100):
    sim.seed(0) # Target is always the same
    target_position = sim.sample_navigable_point()
    sim.seed(seed) # Start depends on the seed
    for _retry in range(number_retries_per_target):
        source_position = sim.sample_navigable_point()
        is_compatible, _ = is_compatible_episode(
            source_position,
            target_position,
            sim,
            near_dist=1,
            far_dist=30,
            geodesic_to_euclid_ratio=1.1,
        )
        if is_compatible:
            break
    if not is_compatible:
        raise ValueError('Cannot find a goal position.')
    return source_position, target_position


def make_gym_env(env_id, seed=0):
    if 'MiniGrid' in env_id:
        env = MiniGridWrapper(gym.make(env_id))
        env.seed(seed)

    elif 'HabitatNav' in env_id:
        config_file = os.path.join(pathlib.Path(__file__).parent.resolve(),
            '..',
            'habitat_config',
            'pointnav_apartment-0.yaml') # Absolute path
        config = get_config(config_paths=config_file,
                opts=['BASE_TASK_CONFIG_PATH', config_file])

        config.defrost()

        # Overwrite all RGBs width / height of TASK (not SIMULATOR)
        for k in config['TASK_CONFIG']['SIMULATOR']:
            if 'rgb' in k.lower():
                config['TASK_CONFIG']['SIMULATOR'][k]['HEIGHT'] = 64
                config['TASK_CONFIG']['SIMULATOR'][k]['WIDTH'] = 64

        # Set Replica scene
        scene = env_id[len('HabitatNav-'):]
        assert len(scene) > 0, 'Undefined scene.'
        config.TASK_CONFIG.DATASET.SCENES_DIR += scene

        config.freeze()

        # Make env
        env_class = get_env_class(config.ENV_NAME)
        env = make_env_fn(env_class=env_class, config=config)

        # Sample and set goal position
        source_location, goal_location = _sample_start_and_goal(env._env._sim, seed)
        env._env._dataset.episodes[0].start_position = source_location # Depends on seed
        env._env._dataset.episodes[0].goals[0].position = goal_location # Fixed

        env = HabitatNavigationWrapper(env)
        env.seed(seed)

    else:
        raise NotImplementedError('Undefined environment.')
    return env


def make_environment(flags, actor_id=1):
    gym_envs = []
    seed = (flags.run_id + 1) * (actor_id + 1)
    # TODO: Habitat does not support naive multi-env like MiniGrid (check VectorEnv)
    # A workaround is to pass a different env to each actor
    for env_name in flags.env.split(','):
        gym_envs.append(make_gym_env(env_name, seed))

    if 'MiniGrid' in flags.env:
        # If fixed_seed is defined, the env seed will be set at every reset(),
        # resulting in the same grid being generated at every episode
        return EnvironmentMiniGrid(gym_envs, no_task=flags.no_reward, fixed_seed=flags.fixed_seed)
    elif 'HabitatNav' in flags.env:
        # In Habitat, the scene is not randomized, so there is no fixed_seed
        # dictpath is used to save visitation dictionaries (for heatmaps)
        dictpath = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid, str(actor_id) + '.pickle')))
        return EnvironmentHabitat(gym_envs, no_task=flags.no_reward, namefile=dictpath)
    else:
        raise NotImplementedError('Undefined environment.')


# ------------------------------------------------------------------------------
# MiniGrid wrappers
# ------------------------------------------------------------------------------

class MiniGridWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, observation):
        return observation['image']


class EnvironmentMiniGrid:
    def __init__(self, gym_envs, no_task=False, fixed_seed=None):
        self.all_envs = gym_envs
        self.env_iter = itertools.cycle(gym_envs)
        self.gym_env = next(self.env_iter)
        self.episode_return = None
        self.episode_step = None
        self.episode_win = None
        self.interactions = None
        self.interactions_dict = dict()
        self.true_state_count = dict() # Count full observations
        self.fixed_seed = fixed_seed
        self.no_task = no_task

    def render(self, mode='human'):
        self.gym_env.render(mode)

    def get_panorama(self):
        # Use a tmp environment, or its internal step counter will increase
        env = deepcopy(self.gym_env)
        dir = env.agent_dir
        while env.agent_dir != 1:
            env.step(1) # Have the agent point at the same direction
        pano = []
        for _ in range(4):
            frame, *_ = env.step(1)
            pano.append(frame)
        # while env.agent_dir != dir:
        #     env.step(1) # Restore direction
        return np.array(pano)

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        self.episode_win = torch.zeros(1, 1, dtype=torch.int32)
        self.interactions = torch.zeros(1, 1, dtype=torch.int32)
        self.interactions_dict = dict()
        initial_done = torch.tensor(False, dtype=torch.bool).view(1, 1)
        initial_real_done = torch.tensor(False, dtype=torch.bool).view(1, 1)
        if self.fixed_seed is not None:
            self.gym_env.seed(seed=self.fixed_seed)
        initial_frame = _format_observation(self.gym_env.reset())
        initial_pano = _format_observation(self.get_panorama())

        return dict(
            panorama=initial_pano,
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            real_done=initial_real_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            episode_win=self.episode_win,
            interactions=self.interactions,
            visited_states=torch.tensor(len(self.true_state_count)).view(1, 1),
            )

    def step(self, action):
        prev_env_str = str(self.gym_env)
        frame, reward, done, _ = self.gym_env.step(action.item())
        env_str = str(self.gym_env)

        # Count interactions
        if action.item() in [3, 4, 5] and prev_env_str != env_str: # Something changed
            interaction_key = (prev_env_str, env_str)
            if interaction_key not in self.interactions_dict: # New unique change
                self.interactions_dict[interaction_key] = 1
                self.interactions[0][0] = 1
            else: # Change is not unique
                self.interactions_dict[interaction_key] += 1
                self.interactions[0][0] = 0
        else:
            self.interactions[0][0] = 0
        interactions = self.interactions

        # # Count true states (see FullyObsWrapper from gym_minigrid)
        # full_grid = self.gym_env.unwrapped.grid.encode()
        # full_grid[self.gym_env.unwrapped.agent_pos[0]][self.gym_env.unwrapped.agent_pos[1]] = np.array([
        #     OBJECT_TO_IDX['agent'],
        #     COLOR_TO_IDX['red'],
        #     self.gym_env.unwrapped.agent_dir
        # ])
        #
        # true_state_key = tuple(full_grid.flatten())
        # if true_state_key in self.true_state_count:
        #     self.true_state_count[true_state_key] += 1
        # else:
        #     self.true_state_count.update({true_state_key: 1})

        # Prevent multiple positive rewards if no_task (save only one for stats)
        if self.episode_win[0][0] == 1:
            reward = 0

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += reward
        episode_return = self.episode_return

        # Prevent multiple wins (if terminal states are ignored)
        if reward > 0 or self.episode_win[0][0] == 1:
            self.episode_win[0][0] = 1
        else:
            self.episode_win[0][0] = 0
        episode_win = self.episode_win

        # 'done' is used only for statistics and it is the one returned by the gym environment
        #        i.e., it is true for terminal states (goal and deaths) and last steps of an episode
        # 'real_done' is true only for terminal states (pretraining does not have terminal states)
        real_done = reward > 0 # TODO: check for 'death' states
        if self.no_task:
            real_done = False

        if real_done or self.gym_env.step_count >= self.gym_env.max_steps:
            self.gym_env = next(self.env_iter)
            if self.fixed_seed is not None:
                self.gym_env.seed(seed=self.fixed_seed)
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
            self.episode_win = torch.zeros(1, 1, dtype=torch.int32)
            self.interactions = torch.zeros(1, 1, dtype=torch.int32)
            self.interactions_dict.clear()

        frame = _format_observation(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done, dtype=torch.bool).view(1, 1)
        real_done = torch.tensor(real_done, dtype=torch.bool).view(1, 1)
        pano = _format_observation(self.get_panorama())

        return dict(
            panorama=pano, # This is four partial obs (panoramic view)
            frame=frame, # This is a single partial obs (egocentric view)
            reward=reward,
            done=done,
            real_done=real_done,
            episode_return=episode_return,
            episode_step=episode_step,
            episode_win=episode_win,
            interactions=interactions,
            visited_states=torch.tensor(len(self.true_state_count)).view(1, 1),
            )

    def close(self):
        for e in self.all_envs:
            e.close()


# ------------------------------------------------------------------------------
# Habitat wrappers
# ------------------------------------------------------------------------------

class HabitatNavigationWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Discrete(env.action_space.n - 1)
        self.observation_space = self.env.observation_space['rgb']
        self._last_full_obs = None
        self._viewer = None

    def reset(self):
        obs = self.env.reset()
        self._last_full_obs = obs
        return np.asarray(obs['rgb'])

    def get_position(self):
        return self.env._env._sim.get_agent_state().position

    def step(self, action):
        obs, rwd, done, info = self.env.step(**{'action': action + 1})
        self._last_full_obs = obs
        obs = np.asarray(obs['rgb'])
        info.update({'position': self.get_position()})
        return obs, rwd, done, info


class EnvironmentHabitat:
    def __init__(self, gym_envs, no_task=False, namefile=''):
        self.all_envs = gym_envs
        self.env_iter = itertools.cycle(gym_envs)
        self.gym_env = next(self.env_iter)
        self.episode_return = None
        self.episode_step = None
        self.no_task = True
        self.true_state_count = dict() # Count (x,y) position (the true state)
        self.namefile = namefile

    def render(self, mode='rgb_array', dt=10):
        if mode == "rgb_array":
            frame = observations_to_image(
                self.gym_env._last_full_obs, self.gym_env.unwrapped._env.get_metrics()
            )
        else:
            raise ValueError(f"Render mode {mode} not currently supported.")
        if self.gym_env._viewer is None:
            self.gym_env._viewer = ImageViewer(self.gym_env.observation_space[0:2], dt)
        self.gym_env._viewer.display(frame)

    def get_panorama(self):
        state = self.gym_env.env._env.sim.get_agent_state()
        rvec = np.zeros((3,))
        pano = []
        for i in range(4):
            rotation = quaternion.from_rotation_vector(rvec)
            obs = self.gym_env.env._env.sim.get_observations_at(state.position, rotation)
            pano.append(obs['rgb'])
            rvec[1] += 2.0 * np.pi / 4
        return np.array(pano)

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.tensor(False, dtype=torch.bool).view(1, 1)
        initial_real_done = torch.tensor(False, dtype=torch.bool).view(1, 1)
        initial_frame = _format_observation(self.gym_env.reset())
        initial_pano = _format_observation(self.get_panorama())

        return dict(
            panorama=initial_pano,
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            real_done=initial_real_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            episode_win=torch.zeros(1, 1, dtype=torch.int32),
            interactions=torch.zeros(1, 1, dtype=torch.int32),
            visited_states=torch.tensor(len(self.true_state_count)).view(1, 1),
            )

    def step(self, action):
        frame, reward, done, info = self.gym_env.step(action.item())

        # Count true states
        position = np.round(np.round(info['position'], 2) * 20) / 20
        true_state_key = tuple([position[0], position[2]])
        if true_state_key in self.true_state_count:
            self.true_state_count[true_state_key] += 1
        else:
            self.true_state_count.update({true_state_key: 1})

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += reward
        episode_return = self.episode_return

        real_done = done # TODO: depends on Habitat task
        if self.no_task:
            done = self.gym_env.unwrapped._env._elapsed_steps >= self.gym_env.unwrapped._env._max_episode_steps
            real_done = False

        if done:
            self.gym_env = next(self.env_iter)
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

            with open(self.namefile, 'wb') as handle:
                pickle.dump(self.true_state_count, handle, protocol=pickle.HIGHEST_PROTOCOL)

        frame = _format_observation(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done, dtype=torch.bool).view(1, 1)
        real_done = torch.tensor(real_done, dtype=torch.bool).view(1, 1)
        pano = _format_observation(self.get_panorama())

        return dict(
            panorama=pano,
            frame=frame,
            reward=reward,
            done=done,
            real_done=real_done,
            episode_return=episode_return,
            episode_step=episode_step,
            episode_win=torch.zeros(1, 1, dtype=torch.int32),
            interactions=torch.zeros(1, 1, dtype=torch.int32),
            visited_states=torch.tensor(len(self.true_state_count)).view(1, 1),
            )

    def close(self):
        for e in self.all_envs:
            e.close()
            if e._viewer is not None:
                e._viewer.close()
