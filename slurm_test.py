import argparse
import datetime
import itertools
import pprint
import os
import submitit

from test_model import run as runner_main
from test_model import parser as runner_parser

os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--local', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--partition', type=str, default='devlab',
                    choices=['learnfair', 'devlab', 'prioritylab'])


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
        expanded.append(new_args)

    return expanded


args_grid = dict(
   from_env=[
#      'HabitatNav-apartment_0',
      'MiniGrid-DoorKey-8x8-v0',
      'MiniGrid-KeyCorridorS3R3-v0',
      'MultiEnv',
    ],
   to_env=[
#        'HabitatNav-apartment_0',
#        'HabitatNav-apartment_1',
#        'HabitatNav-apartment_2',
#        'HabitatNav-room_0',
#        'HabitatNav-room_1',
#        'HabitatNav-room_2',
#        'HabitatNav-frl_apartment_0',
#        'HabitatNav-hotel_0',
#        'HabitatNav-office_3',
       'MiniGrid-Unlock-v0',
       'MiniGrid-DoorKey-8x8-v0',
       'MiniGrid-KeyCorridorS3R3-v0',
       'MiniGrid-UnlockPickup-v0',
       'MiniGrid-BlockedUnlockPickup-v0',
       'MiniGrid-MultiRoom-N6-v0',
       'MiniGrid-MultiRoom-N12-S10-v0',
       'MiniGrid-ObstructedMaze-1Dlh-v0',
       'MiniGrid-ObstructedMaze-2Dlh-v0',
       'MiniGrid-ObstructedMaze-2Dlhb-v0'
    ],
   logdir=[
        'log/21-10-11/baselines-15-31-34-823340',
    ],
   algs=['cbet','ride','rnd','curiosity','random'],
)


# NOTE params is a shallow merge, so do not reuse values
def make_command(params, unique_id):
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
    print(run_args)
    print()

    job_index += 1
    flags = runner_parser.parse_args(make_command(run_args, uid))

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
        comment='neurips_oral_video_10_18',
        time=1319,
        nodes=1,
        ntasks_per_node=1,
        # job setup
        job_name='cbet-testing',
        mem="4GB",
        cpus_per_task=2,
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
