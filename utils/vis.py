import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from collections import OrderedDict

import torch
import torch.nn.functional as F

import wandb


def vis_pose(dataloader, model, epoch, cfg):
    def stick_posemap_on_frame(frame, posemap):
        img = np.dstack((frame, frame, frame))
        img[:, :, 0] = img[:, :, 0] + posemap[0]
        img[:, :, 1] = img[:, :, 1] + posemap[1]
        img = np.clip(img, 0, 1)
        return img

    device = cfg['training']['device']
    if epoch % cfg['output']['vis_every']:
        return

    vis_n = cfg['output']['vis_n']

    batch_x, batch_y_true = next(iter(dataloader))
    with torch.no_grad():
        batch_y_pred = model(batch_x.to(device))
        if type(batch_y_pred) == OrderedDict:
            batch_y_pred = batch_y_pred['out']
        ph, pw = batch_y_pred.size(-2), batch_y_pred.size(-1)
        h, w = batch_y_true.size(-2), batch_y_true.size(-1)
        if ph != h or pw != w:
            batch_y_pred = F.upsample(
                input=batch_y_pred, size=(h, w), mode='bilinear')

    images = []

    for i, (frame, y_true, y_pred) in enumerate(zip(batch_x, batch_y_true, batch_y_pred)):

        true_np = y_true.numpy()
        pred_np = y_pred.cpu().numpy()

        frame_t1_pre = frame[0]
        frame_t1_post = frame[1]
        frame_t2 = frame[2]

        img_t1pre_true = stick_posemap_on_frame(frame_t1_pre, true_np)
        img_t1post_true = stick_posemap_on_frame(frame_t1_post, true_np)
        img_t2_true = stick_posemap_on_frame(frame_t2, true_np)

        img_t1pre_pred = stick_posemap_on_frame(frame_t1_pre, pred_np)
        img_t1post_pred = stick_posemap_on_frame(frame_t1_post, pred_np)
        img_t2_pred = stick_posemap_on_frame(frame_t2, pred_np)

        images.append(
            np.concatenate((img_t1pre_true, img_t1post_true, img_t2_true, img_t1pre_pred, img_t1post_pred, img_t2_pred),
                           axis=1))

        if i >= vis_n - 1:
            break

    wandb.log({"epoch": epoch, "images": [wandb.Image(i) for i in images]})
