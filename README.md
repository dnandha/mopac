# Model Predictive Actor-Critic Reinforcement Learning


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
