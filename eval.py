import numpy as np
import torch
import random
import gym
import imageio
import lorl_env

from train import init_world_model, get_args


def judge_motion_sawyer(desc, s0, st):
    if desc == "move white mug down ":
        dl = st[9:11] - s0[9:11]
        return dl[1] > 0.02
    elif desc == "move white mug up ":
        dl = st[9:11] - s0[9:11]
        return dl[1] < -0.02
    elif desc == "move white mug left ":
        dl = st[9:11] - s0[9:11]
        return dl[0] > 0.02
    elif desc == "move white mug right ":
        dl = st[9:11] - s0[9:11]
        return dl[0] < -0.02
    elif desc == "move black mug down ":
        dl = st[11:13] - s0[11:13]
        return dl[1] > 0.02
    elif desc == "move black mug up ":
        dl = st[11:13] - s0[11:13]
        return dl[1] < -0.02
    elif desc == "move black mug left ":
        dl = st[11:13] - s0[11:13]
        return dl[0] > 0.02
    elif desc == "move black mug right ":
        dl = st[11:13] - s0[11:13]
        return dl[0] < -0.02
    elif desc == "close drawer ":
        dl = st[14] - s0[14]
        return dl > 0.02
    elif desc == "open drawer ":
        dl = st[14] - s0[14]
        return dl < -0.02
    elif desc == "turn faucet left ":
        dl = st[13] - s0[13]
        return dl > np.pi / 10
    elif desc == "turn faucet right ":
        dl = st[13] - s0[13]
        return dl < -np.pi / 10
    

@torch.no_grad()
def sample_action_seq(world_model, init_obs, lang, args):
    init_obs -= 0.5
    if init_obs.shape[-1] == 3:
        init_obs = init_obs.transpose(0, 3, 1, 2)
    obs = torch.as_tensor(init_obs, device=world_model.device).float()
    nonterm = torch.ones((1, 1), device=world_model.device).float()
    lang_embed = world_model.LangModel([lang])
    obs_embed = world_model.ObsEncoder(obs)
    prev_rssm_state = world_model.RSSM._init_rssm_state(1)
    prev_action = torch.zeros((1, args.action_dim), device=world_model.device).float()
    _, rssm_state = world_model.RSSM.rssm_observe(obs_embed, prev_action, nonterm, prev_rssm_state)
    modelstate = world_model.RSSM.get_model_state(rssm_state)
    action_dist = world_model.ActionModel(torch.cat([modelstate, lang_embed], dim=-1))
    action = action_dist.mode()[0]
    action_preds = [action.cpu().numpy()]

    for i in range(1, args.seq_len):
        rssm_state = world_model.RSSM.rssm_imagine(action, rssm_state, nonterm)
        modelstate = world_model.RSSM.get_model_state(rssm_state)
        action_dist = world_model.ActionModel(torch.cat([modelstate, lang_embed], dim=-1))
        action = action_dist.mode()[0]
        action_preds.append(action.cpu().numpy())
    
    return action_preds
    

@torch.no_grad()
def rollout_in_world_model(world_model, init_obs, actions, lang, args):
    init_obs -= 0.5
    if init_obs.shape[-1] == 3:
        init_obs = init_obs.transpose(0, 3, 1, 2)
    b = init_obs.shape[0]
    obs = torch.as_tensor(init_obs, device=world_model.device).float()
    nonterm = torch.ones((b, 1), device=world_model.device).float()
    obs_embed = world_model.ObsEncoder(obs)
    prev_rssm_state = world_model.RSSM._init_rssm_state(b)
    prev_action = torch.zeros((b, args.action_dim), device=world_model.device).float()
    _, rssm_state = world_model.RSSM.rssm_observe(obs_embed, prev_action, nonterm, prev_rssm_state)
    modelstate = world_model.RSSM.get_model_state(rssm_state)
    action = torch.as_tensor(actions[:, 0], device=world_model.device).float()
    obs_preds = [obs.cpu().numpy()]

    for i in range(1, args.seq_len):
        rssm_state = world_model.RSSM.rssm_imagine(action, rssm_state, nonterm)
        modelstate = world_model.RSSM.get_model_state(rssm_state)
        obs_dist = world_model.ObsDecoder(modelstate)
        obs_preds.append(obs_dist.mean.cpu().numpy())
        action = torch.as_tensor(actions[:, i], device=world_model.device).float()
    
    obs_preds = np.stack(obs_preds, axis=1).transpose(0, 1, 3, 4, 2) # (batch, seq_len, h, w, c)
    obs_preds = ((obs_preds + 0.5)*255.0).clip(0, 255).astype(np.uint8)
    for i in range(b):
        imageio.mimsave(f'{args.save_dir}/{lang.replace(" ", "_")}imagine_{i}.gif', obs_preds[i])


def rolllout_in_real_model(env, init_obs, p0, v0, actions, lang, args):
    for i in range(len(actions)):
        env.reset()
        env.set_state(p0, v0)
        obs = init_obs
        obs_list = [obs]
        actions_ = actions[i]
        for action in actions_:
            obs, _, _, _ = env.step(action.flatten())
            obs_list.append(obs)
        obs_list = np.stack(obs_list, axis=0)
        obs_list = ((obs_list)*255.0).clip(0, 255).astype(np.uint8)
        imageio.mimsave(f'{args.save_dir}/{lang.replace(" ", "_")}real_{i}.gif', obs_list)
    

def main(args=get_args()):
    env = gym.make('LorlEnv-v0')
    args.obs_shape = (3, 64, 64)
    args.action_dim = 5
    args.save_dir = "log/sawyer/RSSM/seed_0&timestamp_24-0113-234045/result2"
    print(f'obs shape: {args.obs_shape}, action_dim: {args.action_dim}')

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    world_model= init_world_model(args)
    world_model.load_model('log/sawyer/RSSM/seed_0&timestamp_24-0113-234045/model/models_130.pth')
    world_model._torch_train(False)

    for i in range(10):
        init_obs = env.reset()[0]
    p0, v0 = env.sim.data.qpos[:].copy(), env.sim.data.qvel[:].copy()
    lang = "move white mug down "
    # lang = "turn faucet right "
    actions = sample_action_seq(world_model, np.expand_dims(init_obs.copy(), 0), lang, args)
    num_candidates = 50
    actions = np.expand_dims(np.concatenate(actions, 0), 0).repeat(repeats=num_candidates, axis=0)
    # add Brownian nocise
    brownian_noise_samples = np.zeros_like(actions)
    for i in range(num_candidates):
        for j in range(args.action_dim):
            random_increments = np.random.randn(actions.shape[1])*0.1
            brownian_noise_samples[i, :, j] = np.cumsum(random_increments)
    actions += brownian_noise_samples
    # noise = np.random.randn(*actions.shape)*0.1
    # actions += noise
    actions = actions.clip(-1, 1)

    rollout_in_world_model(world_model, np.expand_dims(init_obs.copy(), 0).repeat(repeats=num_candidates, axis=0), actions, lang, args)
    rolllout_in_real_model(env, init_obs, p0, v0, actions, lang, args)


if __name__ == "__main__":
    main()