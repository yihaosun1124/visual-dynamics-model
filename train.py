import gym
import numpy as np
import torch

import argparse
import random

from models.rssm import EnsembleRSSM
from models.tssm import EnsembleTSSM
from models.dense import DenseModel, Actor
from models.pixel import ObsDecoder, ObsEncoder
from dynamics import WORLD_MODEL
from dynamics.rssm_world_model import get_parameters
from utils.data import OfflineDataset
from utils.logger import Logger, make_log_dirs
from configs import CONFIG


LATENT_MODEL = {'RSSM': EnsembleRSSM, 'TSSM': EnsembleTSSM}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='TSSM')
    parser.add_argument('--task', type=str, default='dmc_cheetah_run')
    parser.add_argument('--data-path', type=str, default='/home/yihaosun/code/rl/vd4rl/new_vd4rl/main/cheetah_run/expert/64px')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--info', type=str, default=None)

    known_args, _ = parser.parse_known_args()
    for arg_key, default_value in CONFIG[known_args.model_type].items():
        parser.add_argument(f'--{arg_key}', default=default_value, type=type(default_value))
    
    args = parser.parse_args()

    return args


def init_world_model(args):
    ssm_info = args.rssm_info if args.model_type == 'RSSM' else args.tssm_info
    ssm_type = args.rssm_info['rssm_type'] if args.model_type == 'RSSM' else args.tssm_info['tssm_type']
    if ssm_type == 'continuous':
        stoch_size = ssm_info['stoch_size']
    elif ssm_type == 'discrete':
        category_size = ssm_info['category_size']
        class_size = ssm_info['class_size']
        stoch_size = category_size*class_size
    args.modelstate_size = stoch_size + ssm_info['deter_size']
    
    latent_model = LATENT_MODEL[args.model_type](
        args.action_dim, args.obs_embedding_size, \
        ssm_info, args.device
    ).to(args.device)
    ObsEncoder_ = ObsEncoder(args.obs_shape, args.obs_embedding_size, **args.obs_encoder).to(args.device)
    ObsDecoder_ = ObsDecoder(args.obs_shape, args.modelstate_size, **args.obs_decoder).to(args.device)
    RewardModel = DenseModel((1,), args.modelstate_size, **args.reward_model).to(args.device)
    ActionModel = Actor((args.action_dim,), args.modelstate_size, **args.action_model).to(args.device)
    world_list = [latent_model, ObsEncoder_, ObsDecoder_, RewardModel, ActionModel]
    if args.loss_scale['discount'] > 0:
        DiscountModel = DenseModel((1,), args.modelstate_size, **args.discount_model).to(args.device)
        world_list.append(DiscountModel)
    else:
        DiscountModel = None
    model_optimizer = torch.optim.Adam(get_parameters(world_list), args.model_lr, weight_decay=1e-6)

    world_model = WORLD_MODEL[args.model_type](
        latent_model, ObsEncoder_, ObsDecoder_, \
        RewardModel, DiscountModel, ActionModel, model_optimizer, \
        seq_len=args.seq_len, kl_info=args.kl_info, \
        loss_scale=args.loss_scale, grad_clip_norm=args.grad_clip_norm, \
        device=args.device
    )

    return world_model


def main(args=get_args()):
    # create dataset
    args.obs_shape = (3, 64, 64)
    args.action_dim = 6
    print(f'obs shape: {args.obs_shape}, action_dim: {args.action_dim}')
    dataset = OfflineDataset(args.data_path, seq_len=args.seq_len, device=args.device)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # logger
    log_dirs = make_log_dirs(args.task, args.model_type, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "world_model_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    world_model= init_world_model(args)

    # train world model
    world_model.train(dataset, args.model_train_steps, batch_size=args.model_batch_size, logger=logger)


if __name__ == "__main__":
    main()