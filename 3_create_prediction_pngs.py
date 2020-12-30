import os
import cv2
import pickle
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from utils.cmaps import default_cmap
from utils.windows import normalize_data
from utils.inference import center_crop, pad_if_needed

DATA_DIR = r"E:/Data/T1T2_HH/npy"
PNG_FOLDER = r"E:/Data/T1T2_HH/predicted_labels"
FOV = 256

predicted_labels = glob(os.path.join(DATA_DIR, "**/*_points.pickle"), recursive=True)

for predicted_label_path in tqdm(predicted_labels):
    pd = pickle.load(open(predicted_label_path, 'rb'))
    xs_end, ys_end, xs_epi, ys_epi = pd['xs_end'], pd['ys_end'], pd['xs_epi'], pd['ys_epi']
    points_end = np.array([list(zip(xs_end, ys_end))])
    points_epi = np.array([list(zip(xs_epi, ys_epi))])

    npy_path = predicted_label_path.split('.npy')[0] + '.npy'

    npy = np.load(npy_path)
    t1w, t2w, pd, t1, t2 = np.transpose(npy, (2, 0, 1))
    t1_pre = normalize_data(t1, window_centre=1300.0, window_width=1300.0)
    t1_post = normalize_data(t1, window_centre=500.0, window_width=1000.0)
    t2 = normalize_data(t2, window_centre=60.0, window_width=120.0)
    t1w = t1w - t1w.min()
    t1w /= t1w.max()
    t2w = t2w - t2w.min()
    t2w /= t2w.max()
    pd = pd - pd.min()
    pd /= pd.max()
    t1_pre = (t1_pre * 255).astype(np.uint8)
    t1_post = (t1_post * 255).astype(np.uint8)
    t2 = (t2 * 255).astype(np.uint8)
    t1w = (t1w * 255).astype(np.uint8)
    t2w = (t2w * 255).astype(np.uint8)
    pd = (pd * 255).astype(np.uint8)

    t1_t2 = np.dstack((t1w, t2w, pd, t1_pre, t1_post, t2))
    t1_t2_crop, _top_left = center_crop(pad_if_needed(t1_t2, min_height=FOV, min_width=FOV), crop_height=FOV,
                                        crop_width=FOV)

    img_out = []

    for i_channel in (3, 4, 5):
        img = default_cmap(t1_t2_crop[:, :, i_channel])
        cv2.polylines(img, points_end, True, [1, 1, 1])
        cv2.polylines(img, points_epi, True, [1, 1, 1])
        img_out.append(img)

    img_out = (np.hstack(img_out) * 255).astype(np.uint8)
    #plt.imshow(img_out)

    date = os.path.basename(os.path.dirname(os.path.dirname(predicted_label_path)))
    study = os.path.basename(os.path.dirname(predicted_label_path))
    labelname = os.path.basename(predicted_label_path)

    out_path = os.path.join(PNG_FOLDER, f"{date}__{study}__{labelname}.png")
    skimage.io.imsave(out_path, img_out)
