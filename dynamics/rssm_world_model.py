import numpy as np
import torch 
import torch.nn as nn
import os 
import math

from tqdm import tqdm, trange
from collections import defaultdict
from PIL import Image
from typing import Iterable


def get_parameters(modules: Iterable[nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class RSSMWorldModel():
    def __init__(
            self,
            RSSM,
            ObsEncoder,
            ObsDecoder,
            RewardModel,
            DiscountModel,
            ActionModel,
            LangModel,
            model_optimizer,
            seq_len,
            kl_info,
            loss_scale,
            grad_clip_norm,
            device="cpu"
        ):
        
        self.device = device

        self.RSSM = RSSM
        self.ObsEncoder = ObsEncoder
        self.ObsDecoder = ObsDecoder
        self.RewardModel = RewardModel
        self.ActionModel = ActionModel
        self.LangModel = LangModel
        self.world_list = [self.RSSM, self.ObsEncoder, self.ObsDecoder, self.RewardModel, self.ActionModel]
        if DiscountModel is not None:
            self.DiscountModel = DiscountModel
            self.world_list.append(self.DiscountModel)
            self._pcont = True
        else:
            self._pcont = False
        self.model_optimizer = model_optimizer

        self._kl_info = kl_info
        self._seq_len = seq_len
        self._loss_scale = loss_scale
        self._grad_clip_norm = grad_clip_norm

    def _torch_train(self, mode=True):
        for model in self.world_list:
            model.train(mode)

    def _preprocess(self, obs):
        obs = (obs.float() / 255.0) - 0.5
        return obs

    def train(self, dataset, train_steps, valid_ratio=0.1, batch_size=64, logger=None):
        valid_size = min(int(len(dataset)*valid_ratio), 1000)
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len(dataset)-valid_size, valid_size])
        num_epochs = int(train_steps // math.ceil(len(train_dataset) / batch_size))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for e in range(1, num_epochs+1):
            self._torch_train(True)
            for obs, actions, rewards, terms, langs in tqdm(train_loader, desc=f'Epoch #{e}/{num_epochs}'):
                metrics = self.train_batch(obs, actions, rewards, terms, langs)
                for k, v in metrics.items():
                    logger.logkv_mean(k, v)

            if e % 10 == 0:
                self._torch_train(False)
                valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
                for obs, actions, rewards, terms, langs in valid_loader:
                    metrics = self.eval_batch(obs, actions, rewards, terms, langs)
                    for k, v in metrics.items():
                        logger.logkv_mean(k, v)
                obs, actions, rewards, terms, langs = valid_dataset.dataset.sample_batch(batch_size)
                self.visualize(obs, actions, rewards, terms, langs, rollout_length=20, save_dir=logger.result_dir)
                self.save_model(e, logger.checkpoint_dir)

            logger.set_timestep(e)
            logger.dumpkvs(exclude=["policy_training_progress"])

        self.save_model(e, logger.model_dir)
    
    def train_batch(self, obs, actions, rewards, terms, langs):
        """ 
        trains the world model
        """
        train_metrics = {}
        obs = self._preprocess(obs)
        seq_len = rewards.shape[1]
        rewards = rewards.unsqueeze(-1) # (batch, t_len, 1)
        nonterms = (1-terms.float()).unsqueeze(-1) # (batch, t_len, 1)
        lang_embeds = self.LangModel(list(langs))

        embed = self.ObsEncoder(obs)                                         #t to t+seq_len
        prev_rssm_state = self.RSSM._init_rssm_state(obs.shape[0])   
        prior, posterior = self.RSSM.rollout_observation(self._seq_len, embed, actions, nonterms, prev_rssm_state)
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len
        obs_dist = self.ObsDecoder(post_modelstate[:, :-1])                     #t to t+seq_len-1  
        reward_dist = self.RewardModel(post_modelstate[:, :-1])               #t to t+seq_len-1
        action_dist = self.ActionModel(torch.cat([post_modelstate[:, :-1].clone().detach(), lang_embeds.unsqueeze(1).repeat(1, seq_len-1, 1)], dim=-1))                #t to t+seq_len-1

        if self._pcont:
            pcont_dist = self.DiscountModel(post_modelstate[:, :-1])                #t to t+seq_len-1
            pcont_loss = self._pcont_loss(pcont_dist, nonterms[:, 1:])
        
        obs_loss = self._obs_loss(obs_dist, obs[:, :-1])
        reward_loss = self._reward_loss(reward_dist, rewards[:, 1:])
        action_masks = np.array([lang!="do nothing" for lang in langs])
        action_loss = self._action_loss(action_dist, actions[:, 1:], action_masks)
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)

        model_loss = self._loss_scale['kl']*div + self._loss_scale['reward']*reward_loss \
            + self._loss_scale['action']*action_loss + obs_loss
        if self._pcont:
            model_loss += self._loss_scale['pcont']*pcont_loss
        
        self.model_optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_parameters(self.world_list), self._grad_clip_norm)
        self.model_optimizer.step()

        with torch.no_grad():
            prior_ent = torch.mean(prior_dist.entropy())
            post_ent = torch.mean(post_dist.entropy())

        train_metrics['train/model_loss'] = model_loss.item()
        train_metrics['train/kl_loss'] = div.item()
        train_metrics['train/reward_loss'] = reward_loss.item()
        train_metrics['train/obs_loss'] = obs_loss.item()
        train_metrics['train/action_loss'] = action_loss.item()
        train_metrics['train/prior_entropy'] = prior_ent.item()
        train_metrics['train/posterior_entropy'] = post_ent.item()
        train_metrics['max_action'] = action_dist.max().item()
        if self._pcont:
            train_metrics['train/pcont_loss'] = pcont_loss.item()

        return train_metrics

    def _obs_loss(self, obs_dist, obs, mode='log_prob'):
        if mode == 'log_prob':
            obs_loss = -torch.mean(obs_dist.log_prob(obs))
        else:
            obs_loss = ((obs_dist.mean - obs) ** 2).mean()
        return obs_loss
    
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self._kl_info['use_kl_balance']:
            alpha = self._kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self._kl_info['use_free_nats']:
                free_nats = self._kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self._kl_info['use_free_nats']:
                free_nats = self._kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    
    def _reward_loss(self, reward_dist, rewards, mode='log_prob'):
        if mode == 'log_prob':
            reward_loss = -torch.mean(reward_dist.log_prob(rewards))
        else:
            reward_loss = ((reward_dist.mean - rewards) ** 2).mean()
        return reward_loss
    
    def _pcont_loss(self, pcont_dist, nonterms):
        pcont_target = nonterms.float()
        pcont_loss = -torch.mean(pcont_dist.log_prob(pcont_target))
        return pcont_loss
    
    def _action_loss(self, action_dist, actions, action_masks, mode='log_prob'):
        # if mode == 'log_prob':
        #     action_loss = -torch.mean(action_dist.log_prob(actions)[action_masks])
        # else:
        #     action_loss = ((action_dist.mode()[0] - actions) ** 2)[action_masks].mean()
        action_loss = ((action_dist - actions) ** 2)[action_masks]
        return action_loss.mean()
    
    @torch.no_grad()
    def eval_batch(self, obs, actions, rewards, terms, langs):
        eval_metrics = {}
        obs = self._preprocess(obs)
        seq_len = rewards.shape[1]
        rewards = rewards.unsqueeze(-1) # (batch, t_len, 1)
        nonterms = (1-terms.float()).unsqueeze(-1) # (batch, t_len, 1)
        lang_embeds = self.LangModel(list(langs))

        embed = self.ObsEncoder(obs)                                         #t to t+seq_len
        prev_rssm_state = self.RSSM._init_rssm_state(obs.shape[0])   
        prior, posterior = self.RSSM.rollout_observation(self._seq_len, embed, actions, nonterms, prev_rssm_state)
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len
        obs_dist = self.ObsDecoder(post_modelstate[:, :-1])                     #t to t+seq_len-1  
        reward_dist = self.RewardModel(post_modelstate[:, :-1])               #t to t+seq_len-1
        action_dist = self.ActionModel(torch.cat([post_modelstate[:, :-1], lang_embeds.unsqueeze(1).repeat(1, seq_len-1, 1)], dim=-1))                #t to t+seq_len-1

        if self._pcont:
            pcont_dist = self.DiscountModel(post_modelstate[:, :-1])                #t to t+seq_len-1
            pcont_loss = self._pcont_loss(pcont_dist, nonterms[:, 1:])
        
        obs_loss = self._obs_loss(obs_dist, obs[:, :-1], mode='mse')
        reward_loss = self._reward_loss(reward_dist, rewards[:, 1:], mode='mse')
        action_masks = np.array([lang!="do nothing" for lang in langs])
        action_loss = self._action_loss(action_dist, actions[:, 1:], action_masks, mode='mse')

        eval_metrics['eval/obs_mse'] = obs_loss.item()
        eval_metrics['eval/reward_mse'] = reward_loss.item()
        eval_metrics['eval/action_mse'] = action_loss.item()

        return eval_metrics
    
    @torch.no_grad()
    def visualize(self, obs, actions, rewards, terms, langs, rollout_length, save_dir):
        obs = self._preprocess(obs)
        b, seq_len, c, h, w = obs.shape
        nonterms = (1-terms.float()).unsqueeze(-1) # (batch, t_len, 1)
        lang_embeds = self.LangModel(list(langs))
        obs_embed = self.ObsEncoder(obs)
        prev_rssm_state = self.RSSM._init_rssm_state(b)
        _, rssm_state = self.RSSM.rssm_observe(obs_embed[:, 0], actions[:, 0], nonterms[:, 0], prev_rssm_state)
        modelstate = self.RSSM.get_model_state(rssm_state)
        action_dist = self.ActionModel(torch.cat([modelstate, lang_embeds], dim=-1))
        # action = action_dist.mode()[0]
        action = action_dist
        obs_preds = [obs[:,0].cpu().numpy()]
        for i in range(1, rollout_length):
            rssm_state = self.RSSM.rssm_imagine(action, rssm_state, nonterms[:, i])
            modelstate = self.RSSM.get_model_state(rssm_state)
            obs_dist = self.ObsDecoder(modelstate)
            obs_preds.append(obs_dist.mean.cpu().numpy())
            action_dist = self.ActionModel(torch.cat([modelstate, lang_embeds], dim=-1))
            # action = action_dist.mode()[0]
            action = action_dist
        obs_preds = np.concatenate(obs_preds, axis=-1) # (batch, c, h, w*seq_len)
        obs_preds = np.transpose(obs_preds, (0, 2, 3, 1)) # (batch, h, w*seq_len, c)
        obs_preds = ((obs_preds + 0.5)*255.0).clip(0, 255).astype(np.uint8)
        obs_gt = ((obs[:, :rollout_length].cpu().numpy() + 0.5)*255.0).clip(0, 255).astype(np.uint8) # (batch, seq_len, c, h, w)
        obs_gt = np.transpose(obs_gt, (0, 3, 1, 4, 2)).reshape(b, h, w*rollout_length, c) # (batch, h, w*seq_len, c)
        
        for i in range(b):
            concat_obs = np.concatenate([obs_gt[i], obs_preds[i]], axis=0)
            Image.fromarray(concat_obs).save(save_dir+f'/vis_{i}.png')

    def save_model(self, iter, save_dir):
        save_dict = {
            'RSSM': self.RSSM.state_dict(),
            'ObsEncoder': self.ObsEncoder.state_dict(),
            'ObsDecoder': self.ObsDecoder.state_dict(),
            'RewardModel': self.RewardModel.state_dict(),
            'ActionModel': self.ActionModel.state_dict(),
        }
        if self._pcont:
            save_dict['DiscountModel'] = self.DiscountModel.state_dict()
        save_path = os.path.join(save_dir, 'models_%d.pth' % iter)
        torch.save(save_dict, save_path)

    def load_model(self, load_path):
        save_dict = torch.load(load_path, map_location=self.device)
        self.RSSM.load_state_dict(save_dict['RSSM'])
        self.ObsEncoder.load_state_dict(save_dict['ObsEncoder'])
        self.ObsDecoder.load_state_dict(save_dict['ObsDecoder'])
        self.RewardModel.load_state_dict(save_dict['RewardModel'])
        self.ActionModel.load_state_dict(save_dict['ActionModel'])
        if self._pcont:
            self.DiscountModel.load_state_dict(save_dict['DiscountModel'])
        self._torch_train(False)
        print('load model from %s' % load_path)