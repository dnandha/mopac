import numpy as np
import tensorflow as tf
import pdb

class FakeEnv:

    def __init__(self, model, config):
        self.model = model
        self.config = config

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, model_inds, deterministic_obs=False, deterministic_rewards=False):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)

        ensemble_next_obs_means = ensemble_model_means[:,:,1:] + obs
        ensemble_next_obs_stds = np.sqrt(ensemble_model_vars[:,:,1:])

        ensemble_rewards_means = ensemble_model_means[:,:,:1]
        ensemble_rewards_stds = np.sqrt(ensemble_model_vars[:,:,:1])

        if deterministic_obs:
            ensemble_samples_next_obs = ensemble_next_obs_means
        else:
            ensemble_samples_next_obs = ensemble_next_obs_means + np.random.normal(size=ensemble_next_obs_means.shape) * ensemble_next_obs_stds

        if deterministic_rewards:
            ensemble_samples_rewards = ensemble_rewards_means
        else:
            ensemble_samples_rewards = ensemble_rewards_means + np.random.normal(size=ensemble_rewards_means.shape) * ensemble_rewards_stds

        #### choose one model from ensemble
        num_models, batch_size, _ = ensemble_model_means.shape
        batch_inds = np.arange(0, batch_size)

        next_obs = ensemble_samples_next_obs[model_inds, batch_inds]
        next_obs_means = ensemble_next_obs_means[model_inds, batch_inds]
        next_obs_stds = ensemble_next_obs_stds[model_inds, batch_inds]

        rewards = ensemble_samples_rewards[model_inds, batch_inds]
        rewards_means = ensemble_rewards_means[model_inds, batch_inds]
        rewards_stds = ensemble_rewards_stds[model_inds, batch_inds]
        ####

        log_prob_next_obs, dev_next_obs = self._get_logprob(next_obs, ensemble_next_obs_means, ensemble_next_obs_stds)
        log_prob_rewards, dev_rewards = self._get_logprob(rewards, ensemble_rewards_means, ensemble_rewards_stds)

        log_prob = np.concatenate((log_prob_rewards, log_prob_next_obs), axis=-1)
        dev = np.concatenate((dev_rewards, dev_next_obs), axis=-1)

        terminals = self.config.termination_fn(obs, act, next_obs)

        #assert batch_size == next_obs_means.shape[0]
        #batch_size = next_obs_means.shape[0]
        return_means = np.concatenate((rewards_means, terminals, next_obs_means), axis=-1)
        return_stds = np.concatenate((rewards_stds, np.zeros((batch_size,1)), next_obs_stds), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        samples = ensemble_samples[0]

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_ph_fn(obs_ph, act_ph, next_obs)
        info = {}

        return next_obs, rewards, terminals, info

    def close(self):
        pass



