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

        self.pngdir = cfg['data']['pngdir']
        self.width, self.height = cfg['data']['image_dim']
        self.high_res = cfg['data'].get('high_res', False)

        self.n_folds = cfg['training']['n_folds']
        self.mixed_precision = cfg['training'].get('mixed_precision', False)

        self.dates = self.load_dates()
        self.sequences = self.load_sequences()
        self.check_image_sizes()

    def load_dates(self):
        """Get each unique date in the PNG directory and split into train/test using seeding for reproducibility"""

        def get_train_test_for_patient(patient):
            random.seed(patient)
            assert 1 <= self.fold <= self.n_folds, f"Fold should be >= 1 and <= {self.n_folds}, not {self.fold}"
            return 'test' if random.randint(1, self.n_folds) == self.fold else 'train'

        assert self.train_or_test in ('train', 'test')
        images = glob(os.path.join(self.pngdir, f"i_*.png"))
        dates = list({i.split('_')[1].split(' ')[0] for i in images})
        dates = [p for p in dates if get_train_test_for_patient(p) == self.train_or_test]
        return dates

    def load_sequences(self):
        """Get a list of tuples of imagepngs:maskpngs"""
        sequences = []
        for date in self.dates:
            imagepaths = glob(os.path.join(self.pngdir, f"i_{date}_*"))  # Get all images
            for imgpath in imagepaths:  # Check matching mask
                maskpath = imgpath.replace('i_', 'm_')
                if os.path.exists(maskpath):
                    if self.high_res:
                        border = self.cfg['data']['border']
                        high_res_coords = self.get_high_res_coords_for_mask(maskpath, border)
                        sequences.append((imgpath, maskpath, high_res_coords))
                    else:
                        sequences.append((imgpath, maskpath, None))
        print(f"{self.train_or_test.upper():<5} FOLD {self.fold}: Loaded {len(sequences)} over {len(self.dates)} dates")
        return sequences

    @staticmethod
    def get_high_res_coords_for_mask(mask, border):
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
        row_from, row_to = np.min(rows)-border, np.max(rows)+border
        col_from, col_to = np.min(cols)-border, np.max(cols)+border
        return row_from, row_to, col_from, col_to

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        imgpath, maskpath, coords = self.sequences[idx]  # coords will be None if not using high_res mode

        img = skimage.io.imread(imgpath)
        mask = skimage.io.imread(maskpath)

        if self.high_res:
            row_from, row_to, col_from, col_to = coords
            img = img[row_from:row_to, col_from:col_to]
            mask = mask[row_from:row_to, col_from:col_to]

        trans = self.transforms(image=img, mask=mask)
        img, mask = trans['image'], trans['mask']

        x = torch.from_numpy(img.transpose([2, 0, 1]))[1:]  # Red colour channel is empty
        y = torch.from_numpy(mask).long()

        if self.mixed_precision:
            x = x.half()
        else:
            x = x.float()

        return x, y

    def get_max_img_dim(self):
        max_height, max_width = 0,0
        for seqs in self.sequences:
            imgpath, maskpath, _coords = seqs
            h,w,c = skimage.io.imread(imgpath).shape
            max_height = max(max_height, h)
            max_width = max(max_width, w)
        return max_height, max_width

    def check_image_sizes(self):
        """Needed if not using high-res, as image will be reflection padded up to this size"""
        max_height, max_width = self.get_max_img_dim()
        assert max_height <= self.height
        assert max_width <= self.width

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
