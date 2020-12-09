import os
import pickle
import numpy as np
from glob import glob

DATA_DIR = r"E:/Data/T1T2_HH/"

predicted_labels = glob(os.path.join(DATA_DIR, "**/*_points.pickle"), recursive=True)

for predicted_label_path in predicted_labels:
    pd = pickle.load(predicted_label_path)
    xs_end, ys_end, xs_epi, ys_epi = pd['xs_end'], pd['ys_end'], pd['xs_epi'], pd['ys_epi']
    points_end = np.array([list(zip(xs_end, ys_end))])
    points_epi = np.array([list(zip(xs_epi, ys_epi))])

    npy_path = predicted_label_path.split('.npy')[0]+'.npy'

