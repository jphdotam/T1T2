import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from glob import glob

import torch

from lib.cfg import load_config
from lib.hrnet import get_hrnet_model, get_hrnet_cfg
from lib.landmarks import load_landmark_model, perform_cmr_landmark_detection, extend_landmarks
from lib.tracing import get_epi_end_paths_from_heatmap_and_landmarks as get_paths
from lib.inference import center_crop, pad_if_needed, get_normalized_channel_stack, prep_normalized_stack_for_inference, \
    tta, paths_to_ridge_polygons
from lib.vis import compute_bullseye_sector_mask_for_slice
from lib.dataset import load_npy_file
from lib.windows import normalize_data
from lib.cmaps import default_cmap

CONFIG = "experiments/036_mini.yaml"
POSE_MODELPATH = r"E:\Dropbox\Work\Other projects\T1T2\output\models\036_mini\70_0.0010686.pt"
# CONFIG = "experiments/030.yaml"
# POSE_MODELPATH = r"E:\Dropbox\Work\Other projects\T1T2\output\models\030\154_0.0004970.pt"
LANDMARK_MODELPATH = "./data/models/landmark_model.pts"
TEST_DICOM_DIR = r"E:\Dropbox\Work\Other projects\T1T2\data\dicoms\mini_test"
FOV = 256
DEVICE = "cuda"

cfg, _ = load_config(CONFIG)

dates_for_studies = glob(os.path.join(TEST_DICOM_DIR, "**/*.npy"), recursive=True)
dates_for_studies = {os.path.basename(os.path.dirname(f)): os.path.basename(os.path.dirname(os.path.dirname(f))) for f
                     in dates_for_studies}

# LOAD MODELS
model = get_hrnet_model(get_hrnet_cfg(cfg)).to(DEVICE)
model = model.eval()
model.load_state_dict(torch.load(POSE_MODELPATH)['state_dict'])

# OUTLIERS = ['T1T2_141613_54120998_54121006_116_20201113-103051__T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy',
#            'T1T2_141613_54120998_54121006_116_20201113-103051__T1_T2_PD_SLC2_CON0_PHS0_REP0_SET0_AVE0_3.npy',
#            'T1T2_141316_22451303_22451310_237_20201118-114441__T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy']

OUTLIERS = ['T1T2_141613_50451101_50451109_342_20201110-130053__T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy',
            'T1T2_141316_12997720_12997727_307_20201109-115256__T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy',
            'T1T2_141613_46420590_46420598_374_20201105-125731__T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy',
            'T1T2_141316_9163956_9163963_427_20201106-152602__T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy',
            'T1T2_141316_7967828_7967835_263_20201105-112420__T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy',
            'T1T2_141316_12997805_12997812_572_20201109-155800__T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy',
            'T1T2_141316_12997720_12997727_307_20201109-115256__T1_T2_PD_SLC2_CON0_PHS0_REP0_SET0_AVE0_3.npy',
            'T1T2_141613_47862334_47862342_176_20201106-104622__T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy',
            'T1T2_141316_7967782_7967789_58_20201105-084313__T1_T2_PD_SLC1_CON0_PHS0_REP0_SET0_AVE0_2.npy',
            'T1T2_141316_12997851_12997858_828_20201109-183708__T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy']

landmark_model = load_landmark_model(LANDMARK_MODELPATH)


def plot_predictions(t1_pre, t1_post, t2, pred, mask_lvcav, mask_lvwall, landmark_points=None):
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))

    # Native
    axes[0, 0].imshow(t1_pre, cmap=default_cmap)
    axes[0, 1].imshow(t1_post, cmap=default_cmap)
    axes[0, 2].imshow(t2, cmap=default_cmap)

    axes[1, 0].imshow(np.max(pred, axis=0))
    axes[1, 1].imshow(mask_lvcav + pred[1])
    axes[1, 2].imshow(mask_lvwall + pred[0])

    if landmark_points is not None:
        x_ant = landmark_points[[0, 2], 0]
        y_ant = landmark_points[[0, 2], 1]
        x_post = landmark_points[[1, 2], 0]
        y_post = landmark_points[[1, 2], 1]

        axes[1, 0].plot(x_ant, y_ant)
        axes[1, 0].plot(x_post, y_post)
        axes[1, 1].plot(x_ant, y_ant)
        axes[1, 1].plot(x_post, y_post)
        axes[1, 2].plot(x_ant, y_ant)
        axes[1, 2].plot(x_post, y_post)

    fig.show()


for i, outlier in enumerate(OUTLIERS):
    seq_dir, npy_name = outlier.split('__')
    date_dir = dates_for_studies[seq_dir]

    t1w, t2w, pd, t1_raw, t2_raw = load_npy_file(os.path.join(TEST_DICOM_DIR, date_dir, seq_dir, npy_name))

    t1_pre, t1_post, t2, t1w, t2w, pd, t1_t2 = get_normalized_channel_stack(t1_raw, t2_raw, t1w, t2w, pd,
                                                                            data_stack_format='all')

    x = prep_normalized_stack_for_inference(t1_t2, FOV, as_tensor=True, tensor_device=DEVICE)

    # Landmark detection
    t2w_landmark, _top_left_landmark = center_crop(pad_if_needed(t2w,
                                                                 min_height=FOV,
                                                                 min_width=FOV),
                                                   crop_height=FOV,
                                                   crop_width=FOV)
    landmark_points, landmark_probs = perform_cmr_landmark_detection(t2w_landmark, model=landmark_model)

    if np.any(landmark_points == -1):
        print(f"Skipping {npy_name} - unable to identify all landmarks")
        continue

    landmark_points = extend_landmarks(landmark_points, FOV)

    with torch.no_grad():
        pred = tta(model, x).cpu().numpy()[0]

    (xs_epi, ys_epi), (xs_end, ys_end) = get_paths(pred, landmark_points)
    mask_lvcav, mask_lvwall = paths_to_ridge_polygons(xs_epi, ys_epi, xs_end, ys_end, FOV)

    plot_predictions(t1_pre, t1_post, t2, pred, mask_lvcav, mask_lvwall, landmark_points)

    break
