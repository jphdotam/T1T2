import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.cfg import load_config
from utils.dataset import T1T2Dataset


CONFIG = "../../experiments/seg/radboud/seg_radboud_002.yaml"

if __name__ ==  "__main__":
    fold = 1

    # Load config
    cfg, model_dir, vis_dir = load_config(CONFIG)

    # Data
    train_transforms, test_transforms = get_segmentation_transforms(cfg)
    dataset_train = ProstateBalancedSegDataset(cfg, 'train', train_transforms, fold=fold)
    dataset_test = ProstateBalancedSegDataset(cfg, 'test', test_transforms, fold=fold)
    dataloader_train = DataLoader(dataset_train, cfg['training']['batch_size'], shuffle=True,
                                  num_workers=cfg['training']['n_workers'], pin_memory=True)
    dataloader_test = DataLoader(dataset_test, cfg['training']['batch_size'], shuffle=False,
                                 num_workers=cfg['training']['n_workers'], pin_memory=True)

    # Model
    model, starting_epoch, state, results = load_seg_model(cfg)
    optimizer, scheduler = load_optimizer(model, cfg, state, steps_per_epoch=len(dataloader_train))
    train_criterion, test_criterion = load_criterion(cfg)

    # Train
    writer = get_summary_writer(cfg, fold=fold)
    best_loss, best_path, last_save_path = 1e10, None, None
    n_epochs, vis_every = cfg['training']['n_epochs'], cfg['output']['vis_every']
    for epoch in range(starting_epoch, n_epochs + 1):
        print(f"\nEpoch {epoch} of {n_epochs}")

        train_loss = cycle_seg('train', model, dataloader_train, epoch, train_criterion, optimizer, cfg, scheduler, writer=writer)
        test_loss = cycle_seg('test', model, dataloader_test, epoch, test_criterion, optimizer, cfg, writer=writer)

        # Log results
        results['epoch'].append(epoch)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        save_results(results, cfg)

        # save model if required('all', 'best', or 'improvement')
        state = {'epoch': epoch + 1,
                 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'results': results,
                 'scheduler': scheduler}
        save_path = os.path.join(model_dir, f"{fold}_{epoch}_{test_loss:.07f}.pt")
        best_loss, last_save_path = save_model(state, save_path, test_loss, best_loss, cfg, last_save_path)

        # vis
        vis_seg(dataloader_test, model, epoch, cfg, vis_dir, show=False, writer=writer, save=False)


