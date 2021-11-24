# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import threading
import time
import timeit
import pprint

import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from src.core import file_writer
from src.core import prof
from src.core import vtrace

import src.models as models
import src.losses as losses

from src.utils import get_batch, log, create_buffers, act
from src.init_models_and_states import init_models_and_states


def learn(actor_model,
          learner_model,
          batch,
          initial_agent_state,
          optimizer,
          scheduler,
          flags,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = learner_model(batch, initial_agent_state)

        bootstrap_value = learner_outputs['baseline'][-1]

        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }

        rewards = batch['reward']
        clipped_rewards = torch.clamp(rewards, -1, 1)

        discounts = (1 - batch['real_done'].float()).abs() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        pg_loss = losses.compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * losses.compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * losses.compute_entropy_loss(
            learner_outputs['policy_logits'])

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch['episode_return'][batch['done']]
        episode_wins = batch['episode_win'][batch['done']]
        interactions = batch['interactions']
        visited_states = batch['visited_states']
        episode_lengths = batch['episode_step'][batch['done']]
        state_count_stats = batch['state_count_stats'][-1]
        change_count_stats = batch['change_count_stats'][-1]
        stats = {
            'total_episodes': torch.sum(batch['done'].float()).item(),
            'episode_wins': torch.sum(episode_wins).item() / flags.batch_size,
            'interactions': torch.sum(interactions).item() / flags.batch_size,
            'visited_states': torch.max(visited_states).item(),
            'mean_episode_length': torch.mean(episode_lengths.float()).item(),
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'state_counts': torch.mean(state_count_stats[:,0]).item(),
            'change_counts': torch.mean(change_count_stats[:,0]).item(),
            'state_counts_std': torch.mean(state_count_stats[:,1]).item(),
            'change_counts_std': torch.mean(change_count_stats[:,1]).item(),
        }

        scheduler.step()
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(learner_model.parameters(), flags.max_grad_norm)
        optimizer.step()

        actor_model.load_state_dict(learner_model.state_dict())
        return stats


def train(flags):
    if flags.xpid is None:
        flags.xpid = 'vanilla-%s' % time.strftime('%Y%m%d-%H%M%S')
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )

    models_and_states = init_models_and_states(flags)
    actor_model = models_and_states['actor_model']
    learner_model = models_and_states['learner_model']
    initial_agent_state_buffers = models_and_states['initial_agent_state_buffers']
    learner_model_optimizer = models_and_states['learner_model_optimizer']
    scheduler = models_and_states['scheduler']
    buffers = models_and_states['buffers']

    reset_state_count_dict = dict()
    state_count_dict = dict()
    reset_change_count_dict = dict()
    change_count_dict = dict()

    actor_processes = []
    ctx = mp.get_context(flags.mp_start)
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, actor_model, buffers,
                  reset_state_count_dict, state_count_dict,
                  reset_change_count_dict, change_count_dict,
                  initial_agent_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    logger = logging.getLogger('logfile')

    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state, exploration_state = get_batch(free_queue, full_queue, buffers,
                initial_agent_state_buffers, flags, timings)
            stats = learn(actor_model, learner_model,
                          batch, agent_state,
                          learner_model_optimizer,
                          scheduler, flags)
            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stats.keys()})
                plogger.log(to_log)
                frames += flags.unroll_length * flags.batch_size

        if i == 0:
            log.info('Batch and learn: %s', timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        checkpointpath = os.path.expandvars(os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))
        log.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'actor_model_state_dict': actor_model.state_dict(),
            'learner_model_optimizer_state_dict': learner_model_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats[
                    'mean_episode_return']
            else:
                mean_return = ''
            total_loss = stats.get('total_loss', float('inf'))
            log.info('After %i frames: loss %f @ %.1f fps. %sStats:\n%s',
                         frames, total_loss, fps, mean_return,
                         pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)
    checkpoint(frames)
    plogger.close()
