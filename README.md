# Robust object grasping in clutter via singulation
This repository is an implementation of the paper 'Robust object grasping in clutter via singulation' in PyBullet.
<p align="center">
    <img src="images/real.gif" height=220px align="center" />
</p>

## Installation
```shell
git clone git@github.com:mkiatos/dqn-singulation.git
cd dqn-singulation

virtualenv ./venv --python=python3
source ./venv/bin/activate
pip install -r requirements.txt
```

Install PytTorch 1.9.0
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Download and install the core code used for tabletop manipulation tasks:
```shell
git clone git@github.com:robot-clutter/clutter_core.git
cd clutter_core
pip install -e .
cd ..
```

## Quick Demo
This demo runs our pre-trained model with a UR5 robot arm in simulation. The objective is to singulate the target object (red one) from its surrounding clutter.
```commandline
python run.py --is_testing --test_trials 10 --episode_max_steps 10 --seed 100
```

## Training
To train the dqn agent from scratch in simulation run the following command:
```commandline
python run.py --n_episodes 10000 --episode_max_steps 10 --save_every 100 --seed 0
```

## Evaluation
To test your own trained model, simply change the location of --checkpoint:
```commandline
python run.py --is_testing --checkpoint checkpoint --test_trials 100 --episode_max_steps 10 --seed 1
```

## Citing
If you find this code useful in your work, please consider citing:
```shell
@inproceedings{kiatos2019robust,
  title={Robust object grasping in clutter via singulation},
  author={Marios, Kiatos and Sotiris, Malassiotis},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={1596--1600},
  year={2019},
  organization={IEEE}
}
```

Note that this is a work in progress.