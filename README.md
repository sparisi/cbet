<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

---

Implementation of **Change-Based Exploration Transfer (C-BET)**, as presented
in [Interesting Object, Curious Agent: Learning Task-Agnostic Exploration](https://arxiv.org/abs/2111.13119).

This code was built on the [RIDE repository](https://github.com/facebookresearch/impact-driven-exploration).


## Codebase and MiniGrid Installation
```
conda create -n cbet python=3.8.10
conda activate cbet
git clone git@github.com:sparisi/cbet.git
cd cbet
pip install -r requirements.txt
```

## Habitat Installation (not Needed for MiniGrid Experiments)
* Follow the [official guide](https://github.com/facebookresearch/habitat-lab/#installation)
and do a **full install** with `habitat_baselines`.
* Download and extract Replica scenes in the root folder of `cbet`
> WARNING! The dataset is very large!

```
sudo apt-get install pigz
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
./download.sh replica-path
```
> If the script does not work, manually unzip with
> `cat replica_v1_0.tar.gz.part* | tar -xz`


## How to Run Experiments
* Intrinsic-only pre-training:
`OMP_NUM_THREADS=1 python main.py --model cbet --env <ENV_NAME> --no_reward --intrinsic_reward_coef=0.005`

* Extrinsic-only transfer with pre-trained model:
`OMP_NUM_THREADS=1 python main.py --model cbet --env <ENV_NAME> --intrinsic_reward_coef=0.0 --checkpoint=path/to/model.tar`
Pre-trained exploration policies are available as [code release](https://github.com/sparisi/cbet/releases/tag/v1).

* Tabula-rasa training with summed intrinsic and extrinsic reward:
`OMP_NUM_THREADS=1 python main.py --model cbet --env <ENV_NAME> --intrinsic_reward_coef=0.005`

See `src/arguments.py` for the full list of hyperparameters.

For MiniGrid, `<ENV_NAME>` can be `MiniGrid-DoorKey-8x8-v0`, `MiniGrid-Unlock-v0`, ...
<br/>
For Habitat, `<ENV_NAME>` can be `HabitatNav-apartment_0`, `HabitatNav-hotel_0`, ...
