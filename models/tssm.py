import numpy as np
import torch
import torch.nn as nn
import transformers
from models.dense import EnsembleLinear
from models.gpt2 import GPT2Model
from utils.tssm import TSSMUtils, TSSMContState, TSSMDiscState


ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'leakyrelu': nn.LeakyReLU,
    'elu': nn.ELU,
    'none': nn.Identity,
}


class EnsembleTSSM(nn.Module, TSSMUtils):
    def __init__(
        self,
        action_size,
        obs_embedding_size,
        tssm_info,
        device,
    ):
        nn.Module.__init__(self)
        TSSMUtils.__init__(self, tssm_type=tssm_info['tssm_type'], info=tssm_info)

        self.device = device

        self.action_size = action_size
        self.obs_embedding_size = obs_embedding_size
        self.num_ensemble = tssm_info['num_ensemble']
        self.hidden_size = tssm_info['hidden_size']
        self.act_fn = ACTIVATIONS[tssm_info['activation']]
        
        # transformer init
        gpt_config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=self.deter_size,
            n_layer=tssm_info['gpt_config']['n_layer'],
            n_head=tssm_info['gpt_config']['n_head'],
            n_inner=4*self.deter_size,
            activation_function=tssm_info['gpt_config']['activation'],
            n_positions=50,
            resid_pdrop=tssm_info['gpt_config']['dropout'],
            attn_pdrop=tssm_info['gpt_config']['dropout'],
        )
        self.transformer = GPT2Model(gpt_config)
        self.embed_timestep = nn.Embedding(50, self.deter_size)
        self.embed_ln = nn.LayerNorm(self.deter_size)

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
        if self.tssm_type == 'discrete':
            temporal_prior += [EnsembleLinear(self.hidden_size, self.stoch_size, self.num_ensemble)]
        elif self.tssm_type == 'continuous':
             temporal_prior += [EnsembleLinear(self.hidden_size, 2 * self.stoch_size, self.num_ensemble)]
        return nn.Sequential(*temporal_prior)

    def _build_temporal_posterior(self):
        """
        model is supposed to take in latest embedded observation
        and output posterior over stochastic states
        omit deterministic state for parallel train
        """
        temporal_posterior = [nn.Linear(self.obs_embedding_size, self.hidden_size)]
        temporal_posterior += [self.act_fn()]
        if self.tssm_type == 'discrete':
            temporal_posterior += [nn.Linear(self.hidden_size, self.stoch_size)]
        elif self.tssm_type == 'continuous':
            temporal_posterior += [nn.Linear(self.hidden_size, 2 * self.stoch_size)]
        return nn.Sequential(*temporal_posterior)
    
    def tssm_imagine(self, prev_actions, prev_tssm_stochs, timesteps=None, attn_masks=None, random_select=True):
        state_action_embeds = self.fc_embed_state_action(torch.cat([prev_tssm_stochs, prev_actions],dim=-1))
        b, seq_len, _ = state_action_embeds.shape

        # transformer forward
        if timesteps is None:
            timesteps = torch.arange(0, seq_len).repeat(b, 1).to(self.device)
        time_embeds = self.embed_timestep(timesteps.long())
        state_action_embeds += time_embeds
        state_action_embeds = self.embed_ln(state_action_embeds)
        if attn_masks is None:
            attn_masks = torch.ones_like(timesteps).long()
        deter_states = self.transformer(inputs_embeds=state_action_embeds, attention_mask=attn_masks)['last_hidden_state'] # (batch, seq_len, dim)

        if self.tssm_type == 'discrete':
            b, seq, dim = deter_states.shape
            prior_logits = self.fc_prior(deter_states.reshape(b*seq, dim))
            prior_logits = prior_logits.reshape(prior_logits.shape[0], b, seq, -1) # (ensemble, batch, seq_len, dim)
            if random_select:
                # random select one from ensemble outputs
                index = np.random.randint(0, self.num_ensemble)
                prior_logits = prior_logits[index]
            else:
                deter_states = deter_states.repeat(self.num_ensemble, 1, 1, 1)
            stats = {'logit':prior_logits}
            prior_stoch_states = self.get_stoch_state(stats)
            prior_tssm_states = TSSMDiscState(prior_logits, prior_stoch_states, deter_states)

        elif self.tssm_type == 'continuous':
            b, seq, dim = deter_states.shape
            prior_means, prior_stds = torch.chunk(self.fc_prior(deter_states.reshape(b*seq, dim)), 2, dim=-1)
            prior_means = prior_means.reshape(prior_means.shape[0], b, seq, -1)
            prior_stds = prior_stds.reshape(prior_stds.shape[0], b, seq, -1)
            if random_select:
                # random select one from ensemble outputs
                index = np.random.randint(0, self.num_ensemble)
                prior_means, prior_stds = prior_means[index], prior_stds[index]
            else:
                deter_states = deter_states.repeat(self.num_ensemble, 1, 1, 1)
            stats = {'mean':prior_means, 'std':prior_stds}
            prior_stoch_states, stds = self.get_stoch_state(stats)
            prior_tssm_states = TSSMContState(prior_means, stds, prior_stoch_states, deter_states)
        return prior_tssm_states

    def tssm_observe(self, obs_embeds: torch.Tensor, actions: torch.Tensor):
        if self.tssm_type == 'discrete':
            posterior_logits = self.fc_posterior(obs_embeds)
            stats = {'logit':posterior_logits}
            posterior_stoch_states = self.get_stoch_state(stats)
            prev_tssm_stoch_states = torch.cat([self._init_tssm_state(obs_embeds.shape[0]).stoch, posterior_stoch_states[:, :-1]], dim=1)
            prior_tssm_states = self.tssm_imagine(actions, prev_tssm_stoch_states)
            posterior_tssm_states = TSSMDiscState(posterior_logits, posterior_stoch_states, prior_tssm_states.deter)
        elif self.tssm_type == 'continuous':
            posterior_means, posterior_stds = torch.chunk(self.fc_posterior(obs_embeds), 2, dim=-1)
            stats = {'mean':posterior_means, 'std':posterior_stds}
            posterior_stoch_states, stds = self.get_stoch_state(stats)
            prev_tssm_stoch_states = torch.cat([self._init_tssm_state(obs_embeds.shape[0]).stoch, posterior_stoch_states[:, :-1]], dim=1)
            prior_tssm_states = self.tssm_imagine(actions, prev_tssm_stoch_states)
            posterior_tssm_states = TSSMContState(posterior_means, stds, posterior_stoch_states, prior_tssm_states.deter)

        return prior_tssm_states, posterior_tssm_states