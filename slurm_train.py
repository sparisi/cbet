import argparse
import datetime
import itertools
import pprint
import os
import submitit
from collections import defaultdict

from main import main as runner_main
from src.arguments import parser as runner_parser

os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--local', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--partition', type=str, default='learnfair',
                    choices=['learnfair', 'devlab', 'prioritylab'])


intrinsic_reward_coef = {
    'vanilla': 0.0,
    'cbet': 0.005,
    'count': 0.005,
    'ride': 0.1,
    'rnd': 0.1,
    'curiosity': 0.1,
}

total_frames = defaultdict(lambda: 50000000)
total_frames.update({
    'HabitatNav-apartment_0': 2000000,
    'MiniGrid-UnlockToy-v0': 10000000,
    'MiniGrid-DoorKey-5x5-v0': 50000000,
    'MiniGrid-LavaCrossingS11N5-v0': 50000000,
    'MiniGrid-Unlock-v0': 5000000,
    'MiniGrid-DoorKey-8x8-v0': 50000000,
    'MiniGrid-KeyCorridorS3R3-v0': 25000000,
    'MiniGrid-UnlockPickup-v0': 25000000,
    'MiniGrid-BlockedUnlockPickup-v0': 50000000,
    'MiniGrid-MultiRoom-N6-v0': 25000000,
    'MiniGrid-MultiRoom-N12-S10-v0': 50000000,
    'MiniGrid-ObstructedMaze-1Dlh-v0': 50000000,
    'MiniGrid-ObstructedMaze-2Dlh-v0': 100000000,
    'MiniGrid-ObstructedMaze-2Dlhb-v0': 200000000,
})



# key => k; some_key => sk
def make_prefix(key):
    tokens = key.split('_')
    return ''.join(w[0] for w in tokens)


def expand_args(params):
    sweep_args = {k: v for k, v in params.items() if isinstance(v, list)}
    # sweep :: [{arg1: val1, arg2: val1}, {arg1: val2, arg2: val2}, ...]
    sweep = [
        dict(zip(sweep_args.keys(), vs))
        for vs in itertools.product(*sweep_args.values())
    ]
    expanded = []
    for swargs in sweep:
        new_args = {**params, **swargs}  # shallow merge
        new_args['xpid'] = '-'.join(
            [f'{make_prefix(k)}{v}' for k, v in swargs.items()])
        expanded.append(new_args)

    for exp in expanded:
        # Depending on your machine, there can be problems between CUDA
        # and EGL when using Habitat. To avoid them, use `spawn`.
        if 'MiniGrid' in exp['env']:
            exp['mp_start'] = 'fork'
        if 'Habitat' in exp['env']:
            exp['mp_start'] = 'spawn'

        # If a checkpoint is passed, we are doing transfer
        if 'checkpoint' in exp and exp['checkpoint']:
            checkpoint_dir, pretrain_env = exp['checkpoint'].split('__')
            if checkpoint_dir[-1] != '/':
                checkpoint_dir += '/'
            runs = os.listdir(checkpoint_dir)
            found = False
            for run in runs:
                rwd = run[run.find('-rt') + 3 :].split('-')[0]
                if 'ri' + str(exp['run_id']) + '-' in run and \
                        ('-m' + exp['model']) in run and \
                        pretrain_env in run:
                    checkpoint_dir += run
                    found = True
                    break
            if not found:
                print('checkpoint NOT found:', exp['checkpoint'])
                exp['checkpoint'] = 'not found'
            else:
                exp['checkpoint'] = os.path.join(checkpoint_dir, 'model.tar')
                print('checkpoint found:', exp['checkpoint'])

            exp['intrinsic_reward_coef'] = 0.

        # If not, it is either pre-train or tabula-rasa
        else:
             exp['intrinsic_reward_coef'] = intrinsic_reward_coef[exp['model']]

        exp['total_frames'] = total_frames[exp['env']]

    return expanded


args_grid = dict(
#    env=[
#      'MiniGrid-DoorKey-8x8-v0',
#      'MiniGrid-KeyCorridorS3R3-v0',
#      'MiniGrid-MultiRoom-N4-S5-v0,MiniGrid-KeyCorridorS3R3-v0,MiniGrid-BlockedUnlockPickup-v0',
#      'MiniGrid-MultiRoomNoisyTV-N7-S4-v0',
#    ],
    env=[
      'MiniGrid-MultiRoom-N6-v0',
      'MiniGrid-BlockedUnlockPickup-v0',
      'MiniGrid-MultiRoom-N12-S10-v0',
      'MiniGrid-ObstructedMaze-1Dlh-v0',
      'MiniGrid-ObstructedMaze-2Dlh-v0',
      'MiniGrid-ObstructedMaze-2Dlhb-v0',
      'MiniGrid-Unlock-v0',
      'MiniGrid-UnlockPickup-v0',
      'MiniGrid-DoorKey-8x8-v0',
      'MiniGrid-KeyCorridorS3R3-v0',
    ],
    run_id=[1,2,3,4,5,6,7,8,9,10],
    num_actors=[40],
    num_buffers=[80], # num_buffers >= 2*num_actors
    unroll_length=[100],
    num_threads=[4],
    batch_size=[32],
    hash_bits=[128],
    discounting=[0.99],
    count_reset_prob=[0.001],
    learning_rate=[0.0001],
    entropy_cost=[0.0005],
    intrinsic_reward_coef=[0.0],
    total_frames=[50000000],
    save_interval=[20],
    checkpoint=[''],
    model=['cbet','count','ride','curiosity','rnd'],#,'vanilla'],
)


# NOTE params is a shallow merge, so do not reuse values
def make_command(params, unique_id):
    params['savedir'] = ('./log/%s/baselines-%s' %
                         (datetime.date.today().strftime('%y-%m-%d'),
                          unique_id))

    # creating cmd-like params
    params = itertools.chain(*[('--%s' % k, str(v))
                               for k, v in params.items()])
    return list(params)


args = parser.parse_args()
args_grid = expand_args(args_grid)
print(f"Submitting {len(args_grid)} jobs to Slurm...")

uid = datetime.datetime.now().strftime('%H-%M-%S-%f')
job_index = 0

for run_args in args_grid:
#    print(run_args)
    print()

    if run_args['checkpoint'] == 'not found':
        print('skipping checkpoint not found')
        continue

    job_index += 1
    flags = runner_parser.parse_args(make_command(run_args, uid))

#    flags.no_reward = True

    print('########## Job {:>4}/{} ##########\nFlags: {}'.format(
        job_index, len(args_grid), flags))

    if args.local:
        executor_cls = submitit.LocalExecutor
    else:
        executor_cls = submitit.SlurmExecutor

    executor = executor_cls(folder='./out/')

    partition = args.partition
    if args.debug:
        partition = 'devlab'

    executor.update_parameters(
        partition=partition,
        comment='neurips_camera_ready_10-26',
        time=4319,
        nodes=1,
        ntasks_per_node=1,
        # job setup
        job_name='cbet_train-%s-%s-%d' % (run_args['model'], run_args['env'], run_args['run_id']),
        mem="32GB", # 64 for Habitat
        cpus_per_task=40,
        num_gpus=1,
        constraint='pascal',
    )

    print('Sending to slurm... ', end='')
    job = executor.submit(runner_main, flags)
    print('Submitted with job id: ', job.job_id)

    if args.debug:
        print('Only running one job on devfair for debugging...')
        print(args)
        import sys
        sys.exit(0)
