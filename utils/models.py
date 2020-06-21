from collections import defaultdict

import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101

import hrnet.lib.models as hrnetmodels
from utils.higher_hrnet import get_2dnet_cfg, get_seg_model

def load_seg_model(cfg, load_model_only=False):
    modeltype = cfg['training']['model']
    if modeltype == 'seg_hrnet':
        model = hrnetmodels.seg_hrnet.get_seg_model(cfg)
    elif modeltype == 'higher_hrnet':
        model = get_seg_model(get_2dnet_cfg())
    elif modeltype == 'fcn_resnet101':
        n_channels = len(cfg['data']['input_classes'])
        n_outputs = len(cfg['data']['output_classes'])
        model = fcn_resnet101(pretrained=False, num_classes=n_outputs)
        model.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif modeltype == 'deeplabv3_resnet101':
        n_channels = len(cfg['data']['input_classes'])
        n_outputs = len(cfg['data']['output_classes'])
        aux_loss = cfg['training']['aux_loss'] is not False
        model = deeplabv3_resnet101(pretrained=False, num_classes=n_outputs, aux_loss=aux_loss)
        model.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        raise ValueError()

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

    model = model.to(cfg['training']['device'])
    if cfg['training']['dataparallel']:
        model = nn.DataParallel(model).to(cfg['training']['device'])

    return model, starting_epoch, state
