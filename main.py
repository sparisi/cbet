from src.arguments import parser

from src.algos.torchbeast import train as train_vanilla
from src.algos.count import train as train_count
from src.algos.curiosity import train as train_curiosity
from src.algos.rnd import train as train_rnd
from src.algos.cbet import train as train_cbet
from src.algos.ride import train as train_ride


def main(flags):
    if flags.model == 'vanilla':
        train_vanilla(flags)
    elif flags.model == 'count':
        train_count(flags)
    elif flags.model == 'curiosity':
        train_curiosity(flags)
    elif flags.model == 'rnd':
        train_rnd(flags)
    elif flags.model == 'ride':
        train_ride(flags)
    elif flags.model == 'cbet':
        train_cbet(flags)
    else:
        raise NotImplementedError("This model has not been implemented. "\
        "The available options are: cbet, vanilla, count, curiosity, rnd, ride.")

if __name__ == '__main__':
    flags = parser.parse_args()
    main(flags)
