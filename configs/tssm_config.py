tssm_config = {
    'tssm_info': {'tssm_type':'discrete', 'num_ensemble':7, 'hidden_size':200, \
                  'deter_size':200, 'stoch_size':30, 'category_size':32, 'class_size':32, 'min_std':0.1, 'activation': 'elu', \
                  'gpt_config': {'embed_size': 200, 'n_layer': 2, 'n_head': 10, 'activation': 'relu', 'dropout': 0.1}},
    'obs_embedding_size': 1024,
    'obs_encoder': {'activation':'elu', 'kernels':[4, 4, 4, 4], 'depth':48},
    'obs_decoder': {'activation':'elu', 'kernels':[5, 5, 6, 6], 'depth':48},
    'reward_model': {'layers':4, 'node_size':400, 'dist':'normal', 'activation':'elu'},
    'discount_model': {'layers':4, 'node_size':400, 'dist':'binary', 'activation':'elu'},
    'action_model': {'layers':4, 'node_size':400, 'activation':'elu'},
    'kl_info': {'use_kl_balance': True, 'kl_balance_scale': 0.8, 'use_free_nats': True, 'free_nats': 1.0},
    'loss_scale': {'kl':1.0, 'reward':1.0, 'discount':0, 'action': 1.0},
    'seq_len': 50,
    'grad_clip_norm': 100,
    'model_lr': 3e-4,
    'model_batch_size': 64,
    'model_train_steps': 25000,
}