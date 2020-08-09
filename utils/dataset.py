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
    def __init__(self, cfg, train_or_test, transforms, fold):
        self.cfg = cfg
        self.train_or_test = train_or_test
        self.transforms = transforms
        self.fold = fold

        self.n_folds = cfg['training']['n_folds']
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
        images = sorted(glob(os.path.join(self.pngdir, f"*__img.png")))
        dates = list({os.path.basename(i).split('__')[0] for i in images})
        dates = [d for d in dates if get_train_test_for_date(d) == self.train_or_test]
        return dates

    def load_sequences(self):
        """Get a list of tuples of (imgpath, labpath)"""
        sequences = []
        for date in self.dates:
            imgpaths = sorted(glob(os.path.join(self.pngdir, f"{date}__*__img.png")))  # Get all images
            for imgpath in imgpaths:  # Check matching mask
                labpath = imgpath.replace('__img', '__lab')
                if os.path.exists(labpath):
                    sequences.append((imgpath, labpath))
        print(f"{self.train_or_test.upper():<5} FOLD {self.fold}: Loaded {len(sequences)} over {len(self.dates)} dates")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        imgpath, labpath = self.sequences[idx]  # coords will be None if not using high_res mode
        n_channels_keep_img = len(self.cfg['export']['sequences'])  # May have exported more channels to make valid PNG
        n_channels_keep_lab = len(self.cfg['export']['label_classes'])

        img = skimage.io.imread(imgpath)[:, :, :n_channels_keep_img]
        lab = skimage.io.imread(labpath)[:, :, :n_channels_keep_lab]
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

    def get_dicom_paths_from_seqences(self, sequence_tuple):
        t1pngpath, t2pngpath, masknpzpath = sequence_tuple
        dicom_root = self.cfg['data']['dicomdir']
        t1pngname = os.path.splitext(os.path.basename(t1pngpath))[0]
        date, study = os.path.basename(t1pngname).split('_', 2)[1:]
        t1paths = glob(os.path.join(dicom_root, date, study, "T1*dcm"))
        t2paths = glob(os.path.join(dicom_root, date, study, "T2*dcm"))
        assert len(t1paths) == 1 and len(t2paths) == 1
        return t1paths[0], t2paths[0]
