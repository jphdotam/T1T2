import os
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from hrnet.lib.core.criterion import CrossEntropy, OhemCrossEntropy


def load_criterion(cfg):
    crit = cfg['training']['criterion']
    class_weights_train = cfg['training'].get('class_weights_train', None)
    class_weights_test = cfg['training'].get('class_weights_test', None)
    ignore_index = cfg['training'].get('ignore_index', -1)

    if class_weights_train:
        print(f"Using class weights {class_weights_train} for training")
        class_weights_train = torch.tensor(class_weights_train).float().to(cfg['training']['device'])
    else:
        class_weights_train = None

    if class_weights_test:
        print(f"Using class weights {class_weights_test} for testing")
        class_weights_test = torch.tensor(class_weights_test).float().to(cfg['training']['device'])
    else:
        class_weights_test = None

    if ignore_index != -1:
        print(f"Ignoring index {ignore_index}")

    def get_criterion(name):
        if name == 'hrnetcrossentropy':
            return CrossEntropy(weight=class_weights_train, ignore_label=ignore_index)
        elif name == 'hrnetomhem':
            return OhemCrossEntropy(weight=class_weights_train, ignore_label=ignore_index)
        elif name == 'crossentropy':
            return nn.CrossEntropyLoss(weight=class_weights_train, ignore_index=ignore_index)
        elif name == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError()

    train_criterion = get_criterion(crit)

    test_crit = cfg['training'].get('test_criterion', False)
    if test_crit:
        test_criterion = get_criterion(test_crit)
    else:
        test_criterion = train_criterion

    return train_criterion, test_criterion


class Am:
    "Simple average meter which stores progress as a running average"

    def __init__(self, n_for_running_average=100):  # n is in samples not batches
        self.n_for_running_average = n_for_running_average
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.running = deque(maxlen=self.n_for_running_average)
        self.running_average = -1

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.running.extend([val] * n)
        self.count += n
        self.avg = self.sum / self.count
        self.running_average = sum(self.running) / len(self.running)


def cycle_seg(train_or_test, model, dataloader, epoch, criterion, optimizer, cfg, scheduler=None, writer=None):
    log_freq = cfg['output']['log_freq']
    device = cfg['training']['device']
    mixed_precision = cfg['training'].get('mixed_precision', False)
    meter_loss = Am()
    accus, accus_nobg = [], []

    if train_or_test == 'train':
        model.train()
        training = True
    elif train_or_test == 'test':
        model.eval()
        training = False
    else:
        raise ValueError(f"train_or_test must be 'train', or 'test', not {train_or_test}")

    for i_batch, (x, y_true) in enumerate(dataloader):
        # Forward pass
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        # Forward pass
        if training:
            if mixed_precision:
                with autocast():
                    y_pred = model(x)
                    loss = criterion(y_pred, y_true)
            else:
                y_pred = model(x)
                loss = criterion(y_pred, y_true)
        else:
            with torch.no_grad():
                if mixed_precision:
                    with autocast():
                        y_pred = model(x)
                        loss = criterion(y_pred, y_true)
                else:
                    y_pred = model(x)
                    loss = criterion(y_pred, y_true)

        # Backward pass
        if training:
            if mixed_precision:
                model.module.scaler.scale(loss).backward()
                model.module.scaler.step(optimizer)
                model.module.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()

        # Metrics
        with torch.no_grad():
            ph, pw = y_pred.size(2), y_pred.size(3)
            h, w = y_true.size(1), y_true.size(2)
            if ph != h or pw != w:
                up_pred = F.upsample(
                    input=y_pred, size=(h, w), mode='bilinear')

            cls_pred = torch.argmax(up_pred, dim=1).flatten()
            cls_true = y_true.flatten()

            accu = (1 - torch.mean((cls_pred == cls_true).float())).cpu().numpy()

            # accu = accuracy_score(cls_true_flat, cls_pred_flat)
            accus.append(accu)

        meter_loss.update(loss, x.size(0))

        # Loss intra-epoch printing
        if (i_batch + 1) % log_freq == 0:

            print(f"{train_or_test.upper(): >5} [{i_batch + 1:04d}/{len(dataloader):04d}]"
                  f"\t\tLOSS: {meter_loss.running_average:.7f}\t\tACCU: {accu:.4f}")

            if writer and training:
                i_iter = ((epoch - 1) * len(dataloader)) + i_batch + 1
                writer.add_scalar(f"LossIter/{train_or_test}", meter_loss.running_average, i_iter)
                writer.add_scalar(f"AccuIter/{train_or_test}", accu, i_iter)

    accu = np.mean(accus)
    print(f"{train_or_test.upper(): >5} Complete!"
          f"\t\t\tLOSS: {meter_loss.avg:.7f}\t\tACCU: {accu:.4f}")

    loss = float(meter_loss.avg.detach().cpu().numpy())

    if writer:
        writer.add_scalar(f"LossEpoch/{train_or_test}", loss, epoch)
        writer.add_scalar(f"AccuEpoch/{train_or_test}", accu, epoch)

    return loss


def save_model(state, save_path, test_metric, best_metric, cfg, last_save_path, lowest_best=True):
    save = cfg['output']['save']
    if save == 'all':
        torch.save(state, save_path)
    elif (test_metric < best_metric) == lowest_best:
        print(f"{test_metric:.5f} better than {best_metric:.5f} -> SAVING")
        if save == 'best':  # Delete previous best if using best only; otherwise keep previous best
            if last_save_path:
                try:
                    os.remove(last_save_path)
                except FileNotFoundError:
                    print(f"Failed to find {last_save_path}")
        best_metric = test_metric
        torch.save(state, save_path)
        last_save_path = save_path
    else:
        print(f"{test_metric:.5g} not improved from {best_metric:.5f}")
    return best_metric, last_save_path
