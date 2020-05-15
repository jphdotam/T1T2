from collections import defaultdict

import torch
import torch.nn as nn

import hrnet.lib.models as hrnetmodels

def load_seg_model(cfg, load_model_only=False):
    model = hrnetmodels.seg_hrnet.get_seg_model(cfg)

    if modelpath := cfg['resume'].get('path', None):
        state = torch.load(modelpath)
        model.load_state_dict(state['state_dict'])
        if load_model_only:
            return model
        starting_epoch = state['epoch']
        if conf_epoch := cfg['resume'].get('epoch', None):
            print(
                f"WARNING: Loaded model trained for {starting_epoch - 1} epochs but config explicitly overrides to {conf_epoch}")
            starting_epoch = conf_epoch
    else:
        if load_model_only:
            print(f"WARNING: Loading model only - assumed for testing but no weights loaded?")
            return model
        starting_epoch = 1
        state = {}

    if cfg['training']['dataparallel']:
        model = nn.DataParallel(model).to(cfg['training']['device'])

    return model, starting_epoch, state
