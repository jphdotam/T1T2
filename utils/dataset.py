import os
import random
import skimage.io
import skimage.measure
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class T1T2Dataset(Dataset):
    def __init__(self, cfg, train_or_test, transforms, fold):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.transforms = transforms
        self.fold = fold

        self.pose_or_seg = cfg['pose_or_seg']
        self.n_folds = cfg['training']['n_folds']
        self.mixed_precision = cfg['training'].get('mixed_precision', False)

        if self.pose_or_seg == 'seg':
            self.pngdir = os.path.join(cfg['data']['pngdir_seg'], cfg['res'])
        elif self.pose_or_seg == 'pose':
            self.pngdir = cfg['data']['pngdir_pose']
        else:
            raise ValueError()

        self.dates = self.load_dates()
        self.sequences = self.load_sequences()

    def load_dates(self):
        """Get each unique date in the PNG directory and split into train/test using seeding for reproducibility"""

        def get_train_test_for_patient(patient):
            random.seed(patient)
            assert 1 <= self.fold <= self.n_folds, f"Fold should be >= 1 and <= {self.n_folds}, not {self.fold}"
            return 'test' if random.randint(1, self.n_folds) == self.fold else 'train'

        assert self.train_or_test in ('train', 'test')
        images = sorted(glob(os.path.join(self.pngdir, f"m_*")))
        dates = list({os.path.basename(i).split('_')[1].split(' ')[0] for i in images})
        dates = [p for p in dates if get_train_test_for_patient(p) == self.train_or_test]
        return dates

    def load_sequences(self):
        if self.pose_or_seg == 'seg':
            return self._load_sequences_seg()
        elif self.pose_or_seg == 'pose':
            return self._load_sequences_pose()
        else:
            raise ValueError()

    def _load_sequences_pose(self):
        """Get a list of tuples of (t1png, t2png, maskpng)"""
        sequences = []
        for date in self.dates:
            t1paths = sorted(glob(os.path.join(self.pngdir, f"t1_{date}_*")))  # Get all images
            for t1path in t1paths:  # Check matching mask
                t2path = t1path.replace('t1_', 't2_')
                maskpath = t1path.replace('t1_', 'm_').replace('.png', '.npz')
                if os.path.exists(maskpath) and os.path.exists(t2path):
                    sequences.append((t1path, t2path, maskpath))
        print(f"{self.train_or_test.upper():<5} FOLD {self.fold}: Loaded {len(sequences)} over {len(self.dates)} dates")
        return sequences

    def _load_sequences_seg(self):
        """Get a list of tuples of (imagepng, maskpng)"""
        sequences = []
        for date in self.dates:
            imagepaths = sorted(glob(os.path.join(self.pngdir, f"i_{date}_*")))  # Get all images
            for imgpath in imagepaths:  # Check matching mask
                maskpath = imgpath.replace('i_', 'm_')
                if os.path.exists(maskpath):
                    sequences.append((imgpath, maskpath))
        print(f"{self.train_or_test.upper():<5} FOLD {self.fold}: Loaded {len(sequences)} over {len(self.dates)} dates")
        return sequences

    @staticmethod
    def get_high_res_coords_for_mask(mask, border=0, get='range'):
        """Can receive an array or a file path"""
        if type(mask) == str:
            mask = skimage.io.imread(mask)
        mask = (mask == 1) | (mask == 2)
        labels = skimage.measure.label(mask)  # Array of trues/falses -> numbered regions
        region_labels, counts = np.unique(labels, return_counts=True)
        largest_label_i = np.argmax(counts[1:]) + 1
        largest_label = region_labels[largest_label_i]
        rows = np.where(np.any(labels == largest_label, axis=1) == True)
        cols = np.where(np.any(labels == largest_label, axis=0) == True)
        if get == 'range':
            row_from, row_to = np.min(rows)-border, np.max(rows)+border
            col_from, col_to = np.min(cols)-border, np.max(cols)+border
            return row_from, row_to, col_from, col_to
        elif get == 'centre':
            assert border == 0
            return int(np.median(rows)), int(np.median(cols))
        else:
            raise ValueError()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.pose_or_seg == 'seg':
            return self._getitem_seg(idx)
        elif self.pose_or_seg == 'pose':
            return self._getitem_pose(idx)
        else:
            raise ValueError()

    def _getitem_pose(self, idx):
        t1path, t2path, maskpath = self.sequences[idx]  # coords will be None if not using high_res mode

        t1 = skimage.io.imread(t1path)  # Mask is float, therefore so should this be
        t2 = skimage.io.imread(t2path)
        mask = (np.load(maskpath, allow_pickle=True)['mask']*255).astype(np.uint8)
        t1t2mask = np.dstack((t1, t2, mask))

        try:
            trans = self.transforms(image=t1t2mask)['image']
        except Exception as e:
            print(f"exception {e} auging {t1t2mask.shape} from {t1path}, {t2path}, {maskpath}: {t1.shape}, {t2.shape}, {mask.shape}")
            raise ValueError()

        t1t2mask = trans.transpose([2, 0, 1])
        img = t1t2mask[:2]
        mask = t1t2mask[2:4]

        # Scale between 0 and 1, as normalisation will have denormalised, and possibly some augs too, e.g. brightness
        mask = (mask - mask.min())
        mask = mask / (mask.max() + 1e-8)

        x = torch.from_numpy(img).float()
        y = torch.from_numpy(mask).float()

        if self.mixed_precision:
            x = x.half()
            y = y.half()
        else:
            x = x.float()
            y = y.float()

        return x, y

    def _getitem_seg(self, idx):
        imgpath, maskpath = self.sequences[idx]  # coords will be None if not using high_res mode

        img = skimage.io.imread(imgpath)
        mask = skimage.io.imread(maskpath)

        trans = self.transforms(image=img, mask=mask)
        img, mask = trans['image'], trans['mask']

        x = torch.from_numpy(img.transpose([2, 0, 1]))[1:]  # Red colour channel is empty
        y = torch.from_numpy(mask).long()

        if self.mixed_precision:
            x = x.half()
        else:
            x = x.float()

        return x, y

    def get_normalisation_params_for_images(self, max=2000):
        mean, std = 0, 0
        print(f"Getting normalisation parameters...")
        print(f"REMEMBER NOT TO HAVE NORMALISATION IN TRANSFORMS (unless you wish to check norm is correct)")
        sequences = self.sequences[:max]
        random.shuffle(sequences)
        for seqs in tqdm(sequences):
            img = skimage.io.imread(seqs[0])

            mask = self.transforms(image=img)['image']

            mask = torch.from_numpy(mask).float().permute(2, 0, 1)[1:]
            mask = mask.view(mask.size(0), -1)  # 3 * all_pixels
            mean += mask.mean(1)
            std += mask.std(1)
        return mean / len(sequences), std / len(sequences)
