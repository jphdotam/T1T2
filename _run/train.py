import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.cfg import load_config
from utils.dataset import T1T2Dataset
from utils.models import load_seg_model
from utils.optimizers import load_optimizer
from utils.transforms import get_segmentation_transforms
from utils.training import load_criterion, cycle_seg, save_model
from utils.tensorboard import get_summary_writer
from utils.vis import vis_seg


CONFIG = "../experiments/020.yaml"

if __name__ ==  "__main__":
    fold = 1

    # Load config
    cfg, vis_dir, model_dir = load_config(CONFIG)

    # Data
    train_transforms, test_transforms = get_segmentation_transforms(cfg)
    ds_train = T1T2Dataset(cfg, 'train', train_transforms, fold)
    ds_test = T1T2Dataset(cfg, 'test', test_transforms, fold)
    dl_train = DataLoader(ds_train, cfg['training']['batch_size'], shuffle=True,
                          num_workers=cfg['training']['num_workers'], pin_memory=True)
    dl_test = DataLoader(ds_test, cfg['training']['batch_size'], shuffle=False,
                         num_workers=1, pin_memory=True)

    # Model
    model, starting_epoch, state = load_seg_model(cfg)
    optimizer, scheduler = load_optimizer(model, cfg, state, steps_per_epoch=len(dl_train))
    train_criterion, test_criterion = load_criterion(cfg)

    # Train
    writer = get_summary_writer(cfg, fold=fold)
    best_loss, best_path, last_save_path = 1e10, None, None
    n_epochs = cfg['training']['n_epochs']
    for epoch in range(starting_epoch, n_epochs + 1):
        print(f"\nEpoch {epoch} of {n_epochs}")

        train_loss = cycle_seg('train', model, dl_train, epoch, train_criterion, optimizer, cfg, scheduler, writer=writer)
        test_loss = cycle_seg('test', model, dl_test, epoch, test_criterion, optimizer, cfg, writer=writer)

        # save model if required('all', 'best', or 'improvement')
        state = {'epoch': epoch + 1,
                 'state_dict': model.module.state_dict() if cfg['training']['dataparallel'] else model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler}
        save_path = os.path.join(model_dir, f"{fold}_{epoch}_{test_loss:.07f}.pt")
        best_loss, last_save_path = save_model(state, save_path, test_loss, best_loss, cfg, last_save_path)

        # vis
        vis_seg(dl_test, model, epoch, vis_dir, cfg, show=False, writer=writer, save=True)

    save_path = os.path.join(model_dir, f"{fold}_final_{n_epochs}_{test_loss:.07f}.pt")
    best_loss, last_save_path = save_model(state, save_path, test_loss, best_loss, cfg, last_save_path, final=True)
