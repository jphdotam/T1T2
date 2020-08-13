import os
import math
import random
import hashlib
import skimage.io
import skimage.measure
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


class T1T2Dataset(Dataset):
    def __init__(self, cfg, train_or_test, transforms, fold=1):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.transforms = transforms
        self.fold = fold

        self.n_folds = cfg['training']['n_folds']
        self.label_format = cfg['export']['format']
        self.mixed_precision = cfg['training'].get('mixed_precision', False)

        self.pngdir = cfg['data']['pngdir']

        self.dates = self.load_dates()
        self.sequences = self.load_sequences()

    def load_dates(self):
        """Get each unique date in the PNG directory and split into train/test using seeding for reproducibility"""

        def get_train_test_for_date(date):
            randnum = int(hashlib.md5(str.encode(date)).hexdigest(), 16) / 16 ** 32
            test_fold = math.floor(randnum * self.n_folds) + 1
            if test_fold == self.fold:
                return 'test'
            else:
                return 'train'

        assert self.train_or_test in ('train', 'test')
        if self.label_format == 'png':
            images = sorted(glob(os.path.join(self.pngdir, f"*__img.png")))
        elif self.label_format == 'npz':
            images = sorted(glob(os.path.join(self.pngdir, f"*__combined.npz")))
        else:
            raise ValueError()
        dates = list({os.path.basename(i).split('__')[0] for i in images})
        dates = [d for d in dates if get_train_test_for_date(d) == self.train_or_test]
        return dates

    def load_sequences(self):
        """Get a list of tuples of (imgpath, labpath)"""
        sequences = []
        for date in sorted(self.dates):
            if self.label_format == 'png':
                imgpaths = sorted(glob(os.path.join(self.pngdir, f"{date}__*__img.png")))  # Get all images
                for imgpath in imgpaths:  # Check matching mask
                    labpath = imgpath.replace('__img', '__lab')
                    if os.path.exists(labpath):
                        sequences.append((imgpath, labpath))
            elif self.label_format == 'npz':
                imgpaths = sorted(glob(os.path.join(self.pngdir, f"{date}__*__combined.npz")))  # Get all images
                sequences.extend(imgpaths)
        print(f"{self.train_or_test.upper():<5} FOLD {self.fold}: Loaded {len(sequences)} over {len(self.dates)} dates")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        n_channels_keep_img = len(self.cfg['export']['sequences'])  # May have exported more channels to make PNG
        n_channels_keep_lab = len(self.cfg['export']['label_classes'])

        if self.label_format == 'png':
            imgpath, labpath = self.sequences[idx]
            img = skimage.io.imread(imgpath)[:, :, :n_channels_keep_img]
            lab = skimage.io.imread(labpath)[:, :, :n_channels_keep_lab]
        elif self.label_format == 'npz':
            imgpath = self.sequences[idx]
            img = np.load(imgpath)['dicom']
            lab = np.load(imgpath)['label']
        else:
            raise ValueError()

        imglab = np.dstack((img, lab))

        trans = self.transforms(image=imglab)['image']

        imglab = trans.transpose([2, 0, 1])
        img = imglab[:n_channels_keep_img]
        lab = imglab[n_channels_keep_img:]

        # BELOW CURRENTLY NOT NEEDED AS WE ARE NOT NORMALISING SO LABELS SHOULD STILL BE VALID
        # Scale between 0 and 1, as normalisation will have denormalised, and possibly some augs too, e.g. brightness
        # lab = (lab - lab.min())
        # lab = lab / (lab.max() + 1e-8)

        x = torch.from_numpy(img).float()
        y = torch.from_numpy(lab).float()

        if self.mixed_precision:
            x = x.half()
            y = y.half()
        else:
            x = x.float()
            y = y.float()

        return x, y

    def get_numpy_paths_for_sequence(self, sequence_tuple):
        npy_root = self.cfg['export']['npydir']
        if self.label_format == 'png':
            imgpath, labpath = sequence_tuple
        elif self.label_format == 'npz':
            imgpath = sequence_tuple
        else:
            raise ValueError()
        datefolder, studyfolder, npyname, _ext = os.path.basename(imgpath).split('__')
        return os.path.join(npy_root, datefolder, studyfolder, npyname + '.npy')
