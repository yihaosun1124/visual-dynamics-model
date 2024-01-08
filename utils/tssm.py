from collections import namedtuple
import torch.distributions as td
import torch
import torch.nn.functional as F
from typing import Union

TSSMDiscState = namedtuple('TSSMDiscState', ['logit', 'stoch', 'deter'])
TSSMContState = namedtuple('TSSMContState', ['mean', 'std', 'stoch', 'deter'])  

TSSMState = Union[TSSMDiscState, TSSMContState]

class TSSMUtils(object):
    '''utility functions for dealing with tssm states'''
    def __init__(self, tssm_type, info):
        self.tssm_type = tssm_type
        if tssm_type == 'continuous':
            self.deter_size = info['deter_size']
            self.stoch_size = info['stoch_size']
            self.min_std = info['min_std']
        elif tssm_type == 'discrete':
            self.deter_size = info['deter_size']
            self.class_size = info['class_size']
            self.category_size = info['category_size']
            self.stoch_size  = self.class_size*self.category_size
        else:
            raise NotImplementedError

    def tssm_seq_to_batch(self, tssm_state, batch_size, seq_len):
        if self.tssm_type == 'discrete':
            return TSSMDiscState(
                seq_to_batch(tssm_state.logit[:, :seq_len], batch_size, seq_len),
                seq_to_batch(tssm_state.stoch[:, :seq_len], batch_size, seq_len),
                seq_to_batch(tssm_state.deter[:, :seq_len], batch_size, seq_len)
            )
        elif self.tssm_type == 'continuous':
            return TSSMContState(
                seq_to_batch(tssm_state.mean[:, :seq_len], batch_size, seq_len),
                seq_to_batch(tssm_state.std[:, :seq_len], batch_size, seq_len),
                seq_to_batch(tssm_state.stoch[:, :seq_len], batch_size, seq_len),
                seq_to_batch(tssm_state.deter[:, :seq_len], batch_size, seq_len)
            )
        
    def tssm_batch_to_seq(self, tssm_state, batch_size, seq_len):
        if self.tssm_type == 'discrete':
            return TSSMDiscState(
                batch_to_seq(tssm_state.logit, batch_size, seq_len),
                batch_to_seq(tssm_state.stoch, batch_size, seq_len),
                batch_to_seq(tssm_state.deter, batch_size, seq_len)
            )
        elif self.tssm_type == 'continuous':
            return TSSMContState(
                batch_to_seq(tssm_state.mean, batch_size, seq_len),
                batch_to_seq(tssm_state.std, batch_size, seq_len),
                batch_to_seq(tssm_state.stoch, batch_size, seq_len),
                batch_to_seq(tssm_state.deter, batch_size, seq_len)
            )
        
    def get_dist(self, tssm_state):
        if self.tssm_type == 'discrete':
            shape = tssm_state.logit.shape
            logit = torch.reshape(tssm_state.logit, shape = (*shape[:-1], self.category_size, self.class_size))
            return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
        elif self.tssm_type == 'continuous':
            return td.independent.Independent(td.Normal(tssm_state.mean, tssm_state.std), 1)

    def get_stoch_state(self, stats):
        if self.tssm_type == 'discrete':
            logit = stats['logit']
            shape = logit.shape
            logit = torch.reshape(logit, shape = (*shape[:-1], self.category_size, self.class_size))
            dist = torch.distributions.OneHotCategorical(logits=logit)
            stoch = dist.sample()
            stoch += dist.probs - dist.probs.detach()
            return torch.flatten(stoch, start_dim=-2, end_dim=-1)

        elif self.tssm_type == 'continuous':
            mean = stats['mean']
            std = stats['std']
            std = F.softplus(std) + self.min_std
            return mean + std*torch.randn_like(mean), std

    def get_model_state(self, tssm_state):
        return torch.cat((tssm_state.deter, tssm_state.stoch), dim=-1)

    def tssm_detach(self, tssm_state):
        if self.tssm_type == 'discrete':
            return TSSMDiscState(
                tssm_state.logit.detach(),  
                tssm_state.stoch.detach(),
                tssm_state.deter.detach(),
            )
        elif self.tssm_type == 'continuous':
            return TSSMContState(
                tssm_state.mean.detach(),
                tssm_state.std.detach(),  
                tssm_state.stoch.detach(),
                tssm_state.deter.detach()
            )

    def _init_tssm_state(self, batch_size, **kwargs):
        if self.tssm_type  == 'discrete':
            return TSSMDiscState(
                torch.zeros(batch_size, 1, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, 1, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, 1, self.deter_size, **kwargs).to(self.device),
            )
        elif self.tssm_type == 'continuous':
            return TSSMContState(
                torch.zeros(batch_size, 1, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, 1, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, 1, self.stoch_size, **kwargs).to(self.device),
                torch.zeros(batch_size, 1, self.deter_size, **kwargs).to(self.device),
            )
            
def seq_to_batch(sequence_data, batch_size, seq_len):
    """
    converts a sequence of batch_size B and length L to a single batch of size B*L
    """
    shp = tuple(sequence_data.shape)
    batch_data = torch.reshape(sequence_data, [shp[0]*shp[1], *shp[2:]])
    return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
    """
    converts a single batch of size B*L to a sequence of batch_size B and length L
    """
    shp = tuple(batch_data.shape)
    seq_data = torch.reshape(batch_data, [batch_size, seq_len, *shp[1:]])
    return seq_data

