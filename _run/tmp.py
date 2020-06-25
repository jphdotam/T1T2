import pydicom
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt

import onnxruntime as ort

from utils.inference import center_crop, dicom_to_img, pad_if_needed, pose_mask_to_coords

# Settings
MEAN = [0.3023, 0.4016]
STD = [0.2582, 0.3189]
FOV_CROP = 256
DOUBLE_INPUT_RES = True
MODELPATH = "../output/models/022/hrnet_pose.onnx"

# Plot settings
RAISE_TO_POWER = 3
NORMALISE_PREDS = True

# def normalize_data(data, WindowCenter, WindowWidth):
#     window_min = max(0, WindowCenter - WindowWidth)
#     frame = data - window_min
#     frame = frame / (WindowWidth)
#     frame = np.clip(frame, 0, 1)
#     frame = frame.astype(np.float32)
#     return frame
#
# t1 = np.load("E:/t1_images_for_AI.npy")
# t2 = np.load("E:/t2_images_for_AI.npy")
#
# t1 = normalize_data(t1, WindowCenter=1300.0, WindowWidth=1300.0)
# t2 = normalize_data(t2, WindowCenter=60.0, WindowWidth=120.0)
#
# t1 = (t1 - MEAN[0]) / STD[0]
# t2 = (t2 - MEAN[1]) / STD[1]
#
# t1, _ = center_crop(pad_if_needed(t1, FOV_CROP, FOV_CROP), FOV_CROP, FOV_CROP)
# t2, _ = center_crop(pad_if_needed(t2, FOV_CROP, FOV_CROP), FOV_CROP, FOV_CROP)
#


# Load data
pred = np.load("E:/t1_t2_images_probs_res.npy")
pred = pred.transpose((3, 2, 0, 1))

for prob_dicom in pred:
    for prob_channel in prob_dicom:
        pred_coords = pose_mask_to_coords(prob_channel)
        pred_coords = [[coord['x'], coord['y']] for coord in pred_coords]
        xs, ys = zip(*pred_coords)
        plt.plot(xs, ys)
    plt.show()

# # Plot
# fig, axes = plt.subplots(2, 2, figsize=(20, 20), gridspec_kw={'wspace': 0, 'hspace': 0})
#
# for i_sequence, sequence in enumerate(('T1', 'T2')):
#     # img
#     img_dcm = img_crop_cl[:, :, i_sequence]
#     img_dcm = np.dstack((img_dcm, img_dcm, img_dcm))
#     img_plot = img_dcm.copy()
#
#     for i_channel in range(2):
#         # pred mask
#         pred = pred_batch[0, i_channel]
#         if RAISE_TO_POWER > 1:
#             pred = np.power(pred, RAISE_TO_POWER)
#         if NORMALISE_PREDS:
#             pred = pred - pred.min()
#             pred = pred / (pred.max() + 1e-8)
#
#         # pred coords
#         pred_coords = pose_mask_to_coords(prediction_mask=pred_batch[0, i_channel])
#         pred_coords = [[coord['x'], coord['y']] for coord in pred_coords]
#         xs, ys = zip(*pred_coords)
#
#         # top row - image & raw predicted masks
#         img_plot[:, :, i_channel] = img_dcm[:, :, i_channel] + pred
#
#         axes[0][i_sequence].imshow(img_plot, cmap='gray')
#         axes[0][i_sequence].set_title(sequence)
#         axes[0][i_sequence].set_axis_off()
#
#         # bottom row - image & interpreted masks
#         axes[1][i_sequence].imshow(img_dcm, cmap='gray')
#         axes[1][i_sequence].plot(xs, ys)
#
# plt.show()
