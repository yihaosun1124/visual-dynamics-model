import numpy as np
import torch
import torch.nn as nn
from models.dense import EnsembleLinear
from utils.rssm import TSSMUtils, RSSMContState, RSSMDiscState


ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'elu': nn.ELU,
    'none': nn.Identity,
}


class EnsembleRSSM(nn.Module, TSSMUtils):
    def __init__(
        self,
        action_size,
        obs_embedding_size,
        rssm_info,
        device,
    ):
        nn.Module.__init__(self)
        TSSMUtils.__init__(self, rssm_type=rssm_info['rssm_type'], info=rssm_info)

        self.device = device

        self.action_size = action_size
        self.obs_embedding_size = obs_embedding_size
        self.num_ensemble = rssm_info['num_ensemble']
        self.hidden_size = rssm_info['hidden_size']
        self.act_fn = ACTIVATIONS[rssm_info['activation']]
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
        self.fc_embed_state_action = self._build_embed_state_action()
        self.fc_prior = self._build_temporal_prior()
        self.fc_posterior = self._build_temporal_posterior()
    
    def _build_embed_state_action(self):
        """
        model is supposed to take in previous stochastic state and previous action 
        and embed it to deter size for rnn input
        """
        fc_embed_state_action = [nn.Linear(self.stoch_size + self.action_size, self.deter_size)]
        fc_embed_state_action += [self.act_fn()]
        return nn.Sequential(*fc_embed_state_action)
    
    def _build_temporal_prior(self):
        """
        model is supposed to take in latest deterministic state 
        and output prior over stochastic state
        """
        temporal_prior = [EnsembleLinear(self.deter_size, self.hidden_size, self.num_ensemble)]
        temporal_prior += [self.act_fn()]
        if self.rssm_type == 'discrete':
            temporal_prior += [EnsembleLinear(self.hidden_size, self.stoch_size, self.num_ensemble)]
        elif self.rssm_type == 'continuous':
             temporal_prior += [EnsembleLinear(self.hidden_size, 2 * self.stoch_size, self.num_ensemble)]
        return nn.Sequential(*temporal_prior)

    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation and deterministic state 
        and output posterior over stochastic states
        """
        temporal_posterior = [nn.Linear(self.deter_size + self.obs_embedding_size, self.hidden_size)]
        temporal_posterior += [self.act_fn()]
        if self.rssm_type == 'discrete':
            temporal_posterior += [nn.Linear(self.hidden_size, self.stoch_size)]
        elif self.rssm_type == 'continuous':
            temporal_posterior += [nn.Linear(self.hidden_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_posterior)
    
    def rssm_imagine(self, prev_action, prev_rssm_state, nonterms=True, random_select=True):
        state_action_embed = self.fc_embed_state_action(torch.cat([prev_rssm_state.stoch*nonterms, prev_action],dim=-1))
        deter_state = self.rnn(state_action_embed, prev_rssm_state.deter*nonterms)
        if self.rssm_type == 'discrete':
            prior_logit = self.fc_prior(deter_state)
            if random_select:
                # random select one from ensemble outputs
                index = np.random.randint(0, self.num_ensemble)
                prior_logit = prior_logit[index]
            else:
                deter_state = deter_state.repeat(self.num_ensemble, 1, 1)
            stats = {'logit':prior_logit}
            prior_stoch_state = self.get_stoch_state(stats)
            prior_rssm_state = RSSMDiscState(prior_logit, prior_stoch_state, deter_state)

        elif self.rssm_type == 'continuous':
            prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
            if random_select:
                # random select one from ensemble outputs
                index = np.random.randint(0, self.num_ensemble)
                prior_mean, prior_std = prior_mean[index], prior_std[index]
            else:
                deter_state = deter_state.repeat(self.num_ensemble, 1, 1)
            stats = {'mean':prior_mean, 'std':prior_std}
            prior_stoch_state, std = self.get_stoch_state(stats)
            prior_rssm_state = RSSMContState(prior_mean, std, prior_stoch_state, deter_state)
        return prior_rssm_state

    def rollout_imagination(self, horizon:int, actor:nn.Module, prev_rssm_state):
        rssm_state = prev_rssm_state
        next_rssm_states = []
        action_entropy = []
        imag_log_probs = []
        for t in range(horizon):
            action, action_dist = actor((self.get_model_state(rssm_state)).detach())
            rssm_state = self.rssm_imagine(action, rssm_state)
            next_rssm_states.append(rssm_state)
            action_entropy.append(action_dist.entropy())
            imag_log_probs.append(action_dist.log_prob(torch.round(action.detach())))

        next_rssm_states = self.rssm_stack_states(next_rssm_states, dim=0)
        action_entropy = torch.stack(action_entropy, dim=0)
        imag_log_probs = torch.stack(imag_log_probs, dim=0)
        return next_rssm_states, imag_log_probs, action_entropy

    def rssm_observe(self, obs_embed, prev_action, prev_nonterm, prev_rssm_state):
        prior_rssm_state = self.rssm_imagine(prev_action, prev_rssm_state, prev_nonterm)
        deter_state = prior_rssm_state.deter
        x = torch.cat([deter_state, obs_embed], dim=-1)
        if self.rssm_type == 'discrete':
            posterior_logit = self.fc_posterior(x)
            stats = {'logit':posterior_logit}
            posterior_stoch_state = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMDiscState(posterior_logit, posterior_stoch_state, deter_state)
        
        elif self.rssm_type == 'continuous':
            posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
            stats = {'mean':posterior_mean, 'std':posterior_std}
            posterior_stoch_state, std = self.get_stoch_state(stats)
            posterior_rssm_state = RSSMContState(posterior_mean, std, posterior_stoch_state, deter_state)
        return prior_rssm_state, posterior_rssm_state

    def rollout_observation(self, seq_len:int, obs_embed: torch.Tensor, action: torch.Tensor, nonterms: torch.Tensor, prev_rssm_state):
        priors = []
        posteriors = []
        for t in range(seq_len):
            prev_action = action[:, t]*nonterms[:, t]
            prior_rssm_state, posterior_rssm_state = self.rssm_observe(obs_embed[:, t], prev_action, nonterms[:, t], prev_rssm_state)
            priors.append(prior_rssm_state)
            posteriors.append(posterior_rssm_state)
            prev_rssm_state = posterior_rssm_state
        prior = self.rssm_stack_states(priors, dim=1)
        post = self.rssm_stack_states(posteriors, dim=1)
        return prior, post