# Robust object grasping in clutter via singulation

<p align="center">
    <img src="images/real.gif" height=220px align="center" />
</p>

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

## Installation
Create a virtual environment and install the package.
```shell
virtualenv ./venv --python=python3
source ./venv/bin/activate
pip install -r requirements.txt
```

Install PytTorch 1.9.0
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## Training
```commandline
```

## Evaluation
To test your own trained model, simply change the location of --snapshot_file:
```commandline
```
