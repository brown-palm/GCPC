# Goal-Conditioned Predictive Coding for Offline Reinforcement Learning

<div align="center">

[[Project page]](https://brown-palm.github.io/GCPC/)
[[arXiv]](https://arxiv.org/abs/2307.03406)
[[Setup]](#Setup)
[[Usage]](#Usage)
[[BibTex]](#Citation)

</div>


## Setup
Install MuJoCo if it is not already the case:
1. Download MuJoCo binary [here](https://github.com/deepmind/mujoco/releases) (mujoco-py requires MuJoCo 2.1.0)
2. Unzip the downloaded archive into `~/.mujoco/`
3. Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`

Install dependencies:
```
conda create -n gcpc python=3.7
codna activate gcpc
pip install -r requirements.txt
pip install -e .
```

To install Antmaze-Ultra, please refer to [link](https://github.com/ZhengyaoJiang/d4rl/)

## Usage
Data preprocessing:
```shell
python -m gcpc.data.parse_d4rl
```

### First Stage: Trajectory Representation Learning

Antmaze
```shell
python -m gcpc.train env_name=antmaze-large-play-v2 model=trajnet exp=antmaze_trl model.model_config.mask_type=mae_rc
```
Kitchen
```shell
python -m gcpc.train env_name=kitchen-mixed-v0 model=trajnet exp=kitchen_trl model.model_config.mask_type=mae_rc
```
Gym
```shell
python -m gcpc.train env_name=halfcheetah-medium-expert-v2 model=trajnet exp=gym_trl model.model_config.mask_type=mae_rc 
```

### Second Stage: Policy Learning

Antmaze
```shell
python -m gcpc.train env_name=antmaze-large-play-v2 model=policynet exp=antmaze_pl model.model_config.tjn_ckpt_path=$tjn_ckpt_path
```
Kitchen
```shell
python -m gcpc.train env_name=kitchen-partial-v0 model=policynet exp=kitchen_pl model.model_config.tjn_ckpt_path=$tjn_ckpt_path
```
Gym
```shell
python -m gcpc.train env_name=halfcheetah-medium-expert-v2 model=policynet exp=gym_pl model.model_config.tjn_ckpt_path=$tjn_ckpt_path
```

To launch policy learning with multiple datasets and seeds, replace the TrajNet checkpoint path in `scripts/launch_pl_<env>.sh` and run
```shell
sh scripts/launch_pl_<env>.sh
```

## Citation
If you find this repository useful for your research, please consider citing our work:

```bibtex
@inproceedings{zeng2023gcpc,
  title={Goal-Conditioned Predictive Coding for Offline Reinforcement Learning},
  author={Zeng, Zilai and Zhang, Ce and Wang, Shijie and Sun, Chen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgements
This repo contains code adapted from [rvs](https://github.com/scottemmons/rvs), [trajectory-transformer](https://github.com/JannerM/trajectory-transformer), [MaskDP_Public](https://github.com/FangchenLiu/MaskDP_public) and [decision-diffuser](https://github.com/anuragajay/decision-diffuser/tree/main/code). We thank the authors and contributors for open-sourcing their code.
