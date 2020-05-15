import os
import random
import skimage.io
from glob import glob

import torch
from torch.utils.data import Dataset


class T1T2Dataset(Dataset):
    def __init__(self, cfg, train_or_test, transforms, fold):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.transforms = transforms
        self.fold = fold

        self.pngdir = cfg['data']['pngdir']
        self.n_folds = cfg['data']['n_folds']
        self.width, height = cfg['data']['image_dim']

        self.mixed_precision = cfg['training'].get('mixed_precision', False)

        self.patients = self.load_patients()
        self.sequences = self.load_sequences()

    def load_patients(self):
        """Get each unique patient in the PNG directory and split into train/test using seeding for reproducibility"""

        def get_train_test_for_patient(patient):
            random.seed(patient)
            assert 1 <= self.fold <= self.n_folds, f"Fold should be >= 1 and <= {self.n_folds}, not {self.fold}"
            return 'test' if random.randint(1, self.n_folds) == self.fold else 'train'

        assert self.train_or_test in ('train', 'test')
        images = os.path.join(self.pngdir, f"i_*.png")
        patients = list({i.split('_')[1].split(' ')[0] for i in images})
        patients = [p for p in patients if get_train_test_for_patient(p) == self.train_or_test]
        return patients

    def load_sequences(self):
        """Get a list of tuples of imagepngs:maskpngs"""
        sequences = []
        for patient in self.patients:
            imagepaths = glob(os.path.join(self.pngdir, f"i_{patient}*"))  # Get all images
            for imgpath in imagepaths:  # Check matching mask
                maskpath = imgpath.replace('i_', 'm_')
                if os.path.exists(maskpath):
                    sequences.append((imgpath, maskpath))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        imgpath, maskpath = self.sequences[idx]

        img = skimage.io.imread(imgpath)
        mask = skimage.io.imread(maskpath)

        trans = self.transforms(image=img, mask=mask)
        img, mask = trans['image'], trans['mask']

        x = torch.from_numpy(img.transpose([2, 0, 1]))
        y = torch.from_numpy(mask).long()

        if self.mixed_precision:
            x = x.half()
        else:
            x = x.float()

        return x, y

    def get_max_img_dim(self):
        max_height, max_width = 0,0
        for seqs in self.sequences:
            imgpath, maskpath = seqs
            h,w,c = skimage.io.imread(imgpath).shape
            max_height = max(max_height, h)
            max_width = max(max_width, w)
        print(f"Max height: {max_height}")
        print(f"Max width: {max_width}")
        return max_height, max_width
