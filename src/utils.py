# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import typing
import gym
import gym_minigrid
import threading
from torch import multiprocessing as mp
import logging
import traceback
import os
import numpy as np
from copy import deepcopy

from src.core import prof
from src.env_utils import make_environment


def _hash_key(x, proj=None, bias=None):
    if proj is None:
        return tuple(x.view(-1).tolist())

    x = x.reshape(1, -1)
    hk = np.dot(x, proj)
    hk = np.tanh(hk)
    hk += bias
    hk = 1 * (hk > 0.5) - 1 * (hk < -0.5)
    return tuple(hk.squeeze().tolist())

def _get_count(k, d):
    return d[k]

def _update_count(k, d):
    if k in d:
        d[k] += 1
    else:
        d.update({k: 1})


shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('torchbeast')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def get_batch(free_queue: mp.SimpleQueue,
              full_queue: mp.SimpleQueue,
              buffers: Buffers,
              agent_state_buffers,
              flags,
              timings,
              exploration_agent_state_buffers=None,
              lock=threading.Lock()):
    with lock:
        timings.time('lock')
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time('dequeue')

    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[agent_state_buffers[m] for m in indices])
    )
    exploration_agent_state = None
    if exploration_agent_state_buffers is not None:
        exploration_agent_state = (
            torch.cat(ts, dim=1)
            for ts in zip(*[exploration_agent_state_buffers[m] for m in indices])
        )
    timings.time('batch')

    for m in indices:
        free_queue.put(m)
    timings.time('enqueue')

    batch = {
        k: t.to(device=flags.device, non_blocking=True)
        for k, t in batch.items()
    }
    agent_state = tuple(t.to(device=flags.device, non_blocking=True)
                                for t in agent_state)
    if exploration_agent_state_buffers:
        exploration_agent_state = tuple(t.to(device=flags.device, non_blocking=True)
                                    for t in exploration_agent_state)
    timings.time('device')

    return batch, agent_state, exploration_agent_state


def create_buffers(obs_shape, num_actions, flags) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        real_done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        episode_win=dict(size=(T + 1,), dtype=torch.int32),
        interactions=dict(size=(T + 1,), dtype=torch.int32),
        visited_states=dict(size=(T + 1,), dtype=torch.int32),
        panorama=dict(size=(T + 1, 4, *obs_shape), dtype=torch.uint8),
        reset_state_count=dict(size=(T + 1, ), dtype=torch.float32),
        state_count=dict(size=(T + 1, ), dtype=torch.float32),
        reset_change_count=dict(size=(T + 1, ), dtype=torch.float32),
        change_count=dict(size=(T + 1, ), dtype=torch.float32),
        reset_sum_count=dict(size=(T + 1, ), dtype=torch.float32),
        sum_count=dict(size=(T + 1, ), dtype=torch.float32),
        state_count_stats=dict(size=(T + 1, 2), dtype=torch.float32), # To log these stats, you must remove dict clear
        change_count_stats=dict(size=(T + 1, 2), dtype=torch.float32), # But this will make the code much much slower
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def act(i: int, free_queue: mp.SimpleQueue, full_queue: mp.SimpleQueue,
        actor_model: torch.nn.Module, buffers: Buffers,
        reset_state_count_dict: dict, state_count_dict: dict,
        reset_change_count_dict: dict, change_count_dict: dict,
        initial_agent_state_buffers, flags,
        exploration_model=None, initial_exploration_agent_state_buffers=None):
    try:
        log.info('Actor %i started.', i)
        timings = prof.Timings()

        env = make_environment(flags, i)
        env_output = env.initial()
        actor_step = 0

        agent_state = actor_model.initial_state(batch_size=1)
        if exploration_model:
            exploration_agent_state = exploration_model.initial_state(batch_size=1)
            exploration_output, unused_state = exploration_model(env_output, exploration_agent_state)
            exploration_logits = exploration_output['policy_logits']
        else:
            exploration_logits = None
        agent_output, unused_state = actor_model(env_output, agent_state, exploration_logits)

        proj_state = None
        bias_state = None
        proj_change = None
        bias_change = None
        if 'Habitat' in flags.env:
            ego_size = np.prod(env_output["frame"].shape)
            pano_size = ego_size * 4
            proj_dim = flags.hash_bits
            proj_state = np.random.normal(0, 1, (proj_dim, ego_size, 1))
            bias_state = np.random.uniform(-1, 1, (proj_dim, 1))
            proj_change = np.random.normal(0, 1, (proj_dim, pano_size, 1))
            bias_change = np.random.uniform(-1, 1, (proj_dim, 1))

        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for key, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][key][...] = tensor
            if exploration_model:
                for key, tensor in enumerate(exploration_agent_state):
                    initial_exploration_agent_state_buffers[index][key][...] = tensor

            state_key = _hash_key(env_output['frame'], proj_state, bias_state)

            # Update state counts (and compute rewards and stats)
            _update_count(state_key, state_count_dict)
            _update_count(state_key, reset_state_count_dict)
            n_s = _get_count(state_key, reset_state_count_dict)
            buffers['reset_state_count'][index][0, ...] = torch.tensor(1 / np.sqrt(n_s))
            n_s = _get_count(state_key, state_count_dict)
            buffers['state_count'][index][0, ...] = torch.tensor(1 / np.sqrt(n_s))
            n_s_stats = [len(state_count_dict), np.std(list(state_count_dict.values()))]
            buffers['state_count_stats'][index][0, ...] = torch.tensor(n_s_stats)

            # If transfer, clear all counts to save memory
            if flags.checkpoint is not None and len(flags.checkpoint) > 0:
                reset_state_count_dict.clear()
                reset_change_count_dict.clear()
                state_count_dict.clear()
                change_count_dict.clear()

            # These do not use counts
            if flags.model in ['vanilla', 'rnd', 'curiosity']:
                reset_state_count_dict.clear()
                reset_change_count_dict.clear()
                state_count_dict.clear()
                change_count_dict.clear()

            # RIDE only uses episodic state counts
            if flags.model == 'ride':
                state_count_dict.clear()
                change_count_dict.clear()
                reset_change_count_dict.clear()

                if env_output['done'][0][0]:
                    reset_state_count_dict.clear()

            # Count only uses state counts without resets
            if flags.model == 'count':
                change_count_dict.clear()
                reset_change_count_dict.clear()
                reset_state_count_dict.clear()

            # Random resets (CBET)
            if np.random.rand() < flags.count_reset_prob and flags.model == 'cbet':
                reset_state_count_dict.clear()
            if np.random.rand() < flags.count_reset_prob and flags.model == 'cbet':
                reset_change_count_dict.clear()

            # At pre-training, C-BET only uses counts with resets
            if flags.model == 'cbet' and flags.no_reward:
                state_count_dict.clear()
                change_count_dict.clear()

            # Do new rollout
            for t in range(flags.unroll_length):
                actor_step += 1

                timings.reset()

                with torch.no_grad():
                    if exploration_model:
                        exploration_output, exploration_agent_state = exploration_model(env_output, exploration_agent_state)
                        exploration_logits = exploration_output['policy_logits']
                    agent_output, agent_state = actor_model(env_output, agent_state, exploration_logits)

                timings.time('actor_model')

                prev_env_output = deepcopy(env_output)
                env_output = env.step(agent_output['action'])

                timings.time('step')

                state_key = _hash_key(env_output['frame'], proj_state, bias_state)
                change_key = _hash_key(env_output['panorama'] - prev_env_output['panorama'], proj_change, bias_change)

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]

                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                # Update counts (no resets) and compute rewards and stats
                _update_count(state_key, state_count_dict)
                _update_count(change_key, change_count_dict)
                n_s = _get_count(state_key, state_count_dict)
                n_c = _get_count(change_key, change_count_dict)
                buffers['state_count'][index][t + 1, ...] = torch.tensor(1 / np.sqrt(n_s))
                buffers['change_count'][index][t + 1, ...] = torch.tensor(1 / np.sqrt(n_c))
                buffers['sum_count'][index][t + 1, ...] = torch.tensor(1 / np.sqrt(n_s + n_c))
                n_s_stats = [len(state_count_dict), np.std(list(state_count_dict.values()))]
                n_c_stats = [len(change_count_dict), np.std(list(change_count_dict.values()))]
                buffers['state_count_stats'][index][t + 1, ...] = torch.tensor(n_s_stats)
                buffers['change_count_stats'][index][t + 1, ...] = torch.tensor(n_c_stats)

                # Update counts (with resets) and compute rewards
                _update_count(state_key, reset_state_count_dict)
                _update_count(change_key, reset_change_count_dict)
                n_s = _get_count(state_key, reset_state_count_dict)
                n_c = _get_count(change_key, reset_change_count_dict)
                buffers['reset_state_count'][index][t + 1, ...] = torch.tensor(1 / np.sqrt(n_s))
                buffers['reset_change_count'][index][t + 1, ...] = torch.tensor(1 / np.sqrt(n_c))
                buffers['reset_sum_count'][index][t + 1, ...] = torch.tensor(1 / np.sqrt(n_s + n_c))

                # If transfer, clear all counts to save memory
                if flags.checkpoint is not None and len(flags.checkpoint) > 0:
                    reset_state_count_dict.clear()
                    reset_change_count_dict.clear()
                    state_count_dict.clear()
                    change_count_dict.clear()

                # These do not use counts
                if flags.model in ['vanilla', 'rnd', 'curiosity']:
                    reset_state_count_dict.clear()
                    reset_change_count_dict.clear()
                    state_count_dict.clear()
                    change_count_dict.clear()

                # RIDE only uses episodic state counts
                if flags.model == 'ride':
                    state_count_dict.clear()
                    change_count_dict.clear()
                    reset_change_count_dict.clear()

                    if env_output['done'][0][0]:
                        reset_state_count_dict.clear()
                        reset_change_count_dict.clear()

                # Count only uses state counts without resets
                if flags.model == 'count':
                    change_count_dict.clear()
                    reset_state_count_dict.clear()
                    reset_change_count_dict.clear()

                # Random resets (CBET)
                if np.random.rand() < flags.count_reset_prob and flags.model == 'cbet':
                    reset_state_count_dict.clear()
                if np.random.rand() < flags.count_reset_prob and flags.model == 'cbet':
                    reset_change_count_dict.clear()

                # At pre-training, C-BET only uses counts with resets
                if flags.model == 'cbet' and flags.no_reward:
                    state_count_dict.clear()
                    change_count_dict.clear()

                timings.time('write')

            full_queue.put(index)

        if i == 0:
            log.info('Actor %i: %s', i, timings.summary())

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
