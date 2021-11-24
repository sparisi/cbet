# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')

# C-BET
parser.add_argument('--count_reset_prob', default=0.001, type=float,
                    help='Counts reset probability at every step during pre-training.')
parser.add_argument('--hash_bits', default=128, type=int,
                    help='Number of bits of hash encodings.')

# General settings
parser.add_argument('--env', type=str, default='MiniGrid-DoorKey-6x6-v0',
                    help='Gym environment. To enter multiple environments trained in parallel, \
                    add them as a comma-separated list.')
parser.add_argument('--xpid', default=None,
                    help='Experiment id (default: None).')
parser.add_argument('--num_input_frames', default=1, type=int,
                    help='Number of input frames to the model and state embedding including the current frame \
                    When num_input_frames > 1, it will also take the previous num_input_frames - 1 frames as input.')
parser.add_argument('--run_id', default=0, type=int,
                    help='Run id used for running multiple instances of the same HP set \
                    (instead of a different random seed since torchbeast does not accept this).')
parser.add_argument('--seed', default=0, type=int,
                    help='Environment seed.')
parser.add_argument('--save_interval', default=10, type=int, metavar='N',
                    help='Time interval (in minutes) at which to save the model.')
parser.add_argument('--checkpoint_num_frames', default=10000000, type=int,
                    help='Number of frames for checkpoint to load.')
parser.add_argument('--checkpoint', default=None,
                    help='Path to model.tar for loading checkpoint from past run.')

# Training settings
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint.')
parser.add_argument('--savedir', default='logs',
                    help='Root dir where experiment data will be saved.')
parser.add_argument('--num_actors', default=40, type=int, metavar='N',
                    help='Number of actors.')
parser.add_argument('--total_frames', default=50000000, type=int, metavar='T',
                    help='Total environment frames to train for.')
parser.add_argument('--batch_size', default=32, type=int, metavar='B',
                    help='Learner batch size.')
parser.add_argument('--unroll_length', default=100, type=int, metavar='T',
                    help='The unroll length (time dimension).')
parser.add_argument('--queue_timeout', default=1, type=int,
                    metavar='S', help='Error timeout for queue.')
parser.add_argument('--num_buffers', default=40, type=int,
                    metavar='N', help='Number of shared-memory buffers.')
parser.add_argument('--num_threads', default=4, type=int,
                    metavar='N', help='Number learner threads.')
parser.add_argument('--mp_start', default='spawn', type=str,
                    metavar='N', help='Start method of multiprocesses. \
                    Depending on your machine, there can be problems between CUDA \
                    and EGL when using Habitat. To avoid them, use `spawn`.')
parser.add_argument('--disable_cuda', action='store_true',
                    help='Disable CUDA.')

# Loss settings
parser.add_argument('--entropy_cost', default=0.0005, type=float,
                    help='Entropy cost/multiplier.')
parser.add_argument('--baseline_cost', default=0.5, type=float,
                    help='Baseline cost/multiplier.')
parser.add_argument('--discounting', default=0.99, type=float,
                    help='Discounting factor.')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='Learning rate.')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant.')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum.')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon.')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    metavar='MGN', help='Max norm of gradients.')

# Exploration settings
parser.add_argument('--forward_loss_coef', default=10.0, type=float,
                    help='Coefficient for the forward dynamics loss. \
                    This weighs the inverse model loss agains the forward model loss. \
                    Should be between 0 and 1.')
parser.add_argument('--inverse_loss_coef', default=0.1, type=float,
                    help='Coefficient for the forward dynamics loss. \
                    This weighs the inverse model loss agains the forward model loss. \
                    Should be between 0 and 1.')
parser.add_argument('--intrinsic_reward_coef', default=0.5, type=float,
                    help='Coefficient for the intrinsic reward. \
                    This weighs the intrinsic reaward against the extrinsic one. \
                    Should be larger than 0.')
parser.add_argument('--rnd_loss_coef', default=0.1, type=float,
                    help='Coefficient for the RND loss coefficient relative to the IMPALA one.')

# Singleton environments
parser.add_argument('--fixed_seed', required=False, type=int,
                    help='Fix the environment seed so that it is \
                    no longer procedurally generated but rather the same layout every episode.')
parser.add_argument('--no_reward', action='store_true',
                    help='No extrinsic reward. The agent uses only intrinsic reward to learn.')

# Training models
parser.add_argument('--model', default='vanilla',
                    choices=['vanilla', 'count', 'curiosity', 'rnd', 'ride', 'cbet'],
                    help='Model used for training the agent.'
                    )
