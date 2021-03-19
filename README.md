# Model Predictive Actor-Critic Reinforcement Learning

<p align="center">
	<img src="https://drive.google.com/uc?export=view&id=1XNVqPc2UdzMWWbHsp0kT40I-BmEngMnk" width="80%">
</p>


## Abstract

Substantial advancements to model-based reinforcement learning algorithms have been impeded by the model-bias induced by the collected data, which generally hurts performance. Meanwhile, their inherent sample efficiency warrants utility for most robot applications, limiting potential damage to the robot and its environment during training. Inspired by information theoretic model predictive control and advances in deep reinforcement learning, we introduce Model Predictive Actor-Critic (MoPAC), a hybrid model-based/model-free method that combines model predictive rollouts with policy optimization as to mitigate model bias. MoPAC leverages optimal trajectories to guide policy learning, but explores via its model-free method, allowing the algorithm to learn more expressive dynamics models. This combination guarantees optimal skill learning up to an approximation error and reduces necessary physical interaction with the environment, making it suitable for real-robot training. We provide extensive results showcasing how our proposed method generally outperforms current state-of-the-art and conclude by evaluating MoPAC for learning on a physical robotic hand performing valve rotation and finger gaiting--a task that requires grasping, manipulation, and then regrasping of an object. 

## Reference

```
@inproceedings{morgan2021model,
  author = {Andrew Morgan and Daljeet Nandha and Georgia Chalvatzaki and Carlo D'Eramo and Aaron Dollar and Jan Peters},
  title = {Model Predictive Actor-Critic: Accelerating Robot Skill Acquisition with Deep Reinforcement Learning},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2021}
}
```

## Installation
1. Install MuJoCo (https://www.roboti.us/index.html) at `~/.mujoco/mujoco200` and copy your license key to `~/.mujoco/mjkey.txt`
2. Clone `mopac`
```
git clone --recursive https://github.com/dnandha/mopac.git
```
3. Create a conda environment and install mopac
```
cd mopac
conda env create -f environment.yml
conda activate mopac
pip install -e .
pip install -e viskit
```

## Usage
Configuration files can be found in [`mopac/examples/config/`](mopac/examples/config).

Running locally:
```
mopac run_local mopac.examples.development --config=mopac.examples.config.${envname}.0 --gpus=1 --trial-gpus=1
```

Running on cluster:
```
ray start --block --head --redis-port=6379 --temp-dir=${ray_tmp_dir} &
mopac run_example_cluster mopac.examples.development --config=mopac.examples.config.${envname}.0
```

Restoring a checkpoint locally:
```
mopac run_local mopac.examples.development --config=mopac.examples.config.${envname}.0 --gpus=1 --trial-gpus=1 --restore=${path_to_checkpoint_ending_with_slash}
```
```
for chkpt in $(find ~/ray_mopac/${env_name}/${exp_name} -name checkpoint_${number}); do mopac run_local mopac.examples.development --config=mopac.examples.config.${envname}.0 --gpus=1 --trial-gpus=1 --restore=${chkpt}/; done
```

Restoring a checkpoint on cluster:
```
for chkpt in $(find ~/ray_mopac/${env_name}/${exp_name} -name checkpoint_${number}); do export chkpt=${chkpt}/; sbatch -J ${path_to_job_script}; done
```
Job sript must contain `=-restore=${chkpt}`, e.g. `mopac run_example_cluster mopac.examples.development --config=mopac.examples.config.${envname}.0 --restore=${chkpt}`.

#### New environments
To run on a different environment, you can modify the provided [template](examples/config/custom/0.py). You will also need to provide the termination function for the environment in [`mopac/static`](mopac/static). If you name the file the lowercase version of the environment name, it will be found automatically. See [`hopper.py`](mopac/static/hopper.py) for an example.

#### Logging

This codebase contains [viskit](https://github.com/vitchyr/viskit) as a submodule. You can view saved runs with:
```
viskit ~/ray_mopac --port 6008
```
assuming you used the default [`log_dir`](mopac/examples/config/halfcheetah/0.py#L7).

#### Hyperparameters

The rollout length schedule is defined by a length-4 list in a [config file](mopac/examples/config/halfcheetah/0.py#L31). The format is `[start_epoch, end_epoch, start_length, end_length]`, so the following:
```
'rollout_schedule': [20, 100, 5, 15] 
```

The mix ratio of model-based and model-free samples is defined by a length-4 list in a [config file](mopac/examples/config/halfcheetah/0.py#L31). The format is `[start_epoch, end_epoch, start_length, end_length]`, so the following:
```
'ratio_schedule': [20, 100, 5, 15] 
```

This corresponds to a model rollout length linearly increasing from 5 to 15 over epochs 20 to 100. 

If you want to speed up training in terms of wall clock time (but possibly make the runs less sample-efficient), you can set a timeout for model training ([`max_model_t`](mopac/examples/config/halfcheetah/0.py#L30), in seconds) or train the model less frequently (every [`model_train_freq`](mopac/examples/config/halfcheetah/0.py#L22) steps).


## Acknowledgments
The underlying soft actor-critic implementation in MOPAC comes from [Tuomas Haarnoja](https://scholar.google.com/citations?user=VT7peyEAAAAJ&hl=en) and [Kristian Hartikainen's](https://hartikainen.github.io/) [softlearning](https://github.com/rail-berkeley/softlearning) codebase. The modeling code is a slightly modified version of [Kurtland Chua's](https://kchua.github.io/) [PETS](https://github.com/kchua/handful-of-trials) implementation.

This code is an extension of [MBPO](https://github.com/JannerM/mbpo) for model predictive rollouts.
