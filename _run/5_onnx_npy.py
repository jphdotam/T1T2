import os
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import onnxruntime as ort

from utils.cfg import load_config
from utils.dataset import T1T2Dataset
from utils.windows import window_numpy
from utils.inference import center_crop, dicom_to_img, pad_if_needed, pose_mask_to_coords
from utils.cmaps import default_cmap

# Settings
SRC = "E:/Data/T1T2_peter/20200427/T1T2_42363_622646506_622646511_675_20200427-163827/T1_T2_PD_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.npy"
PRED_DIR = "../output/predictions_26/"
CONFIG = "../experiments/026.yaml"
MODELPATH = "../output/models/026/110_0.0010794.pt.onnx"
SEQUENCE_WINDOWS = {"t1w": {'channel': 0, 'method':'divmax'},
                    "t2w": {'channel': 1, 'method':'divmax'},
                    "pd": {'channel': 2, 'method':'divmax'},
                    "t1_pre": {'channel': 3, 'method': 'window', 'wc': 1300, 'ww': 1300},
                    "t1_post": {'channel': 3, 'method': 'window', 'wc': 500, 'ww': 1000},
                    "t2": {'channel': 4, 'method': 'window', 'wc': 60, 'ww': 120}}
FOV_CROP = 256
DOUBLE_INPUT_RES = True  # HRNet downsamples 2x, so double upsample ensures predictions are 256 * 256 res

# Plot settings
RAISE_TO_POWER = 1
NORMALISE_PREDS = True

# Load data
cfg, vis_dir, model_dir = load_config(CONFIG)
ds_test = T1T2Dataset(cfg, 'test', None, 1)


def preds_to_rgb(preds):
    """Take predictions (with or without batch dimension) and put each layer into R, G, or B sequentially"""
    if len(preds.shape) == 4:
        preds = preds[0]  # Remove batch dimension
    out = np.zeros((preds.shape[1], preds.shape[2], 3))
    for i_pred_layer, pred_layer in enumerate(preds):
        out[:, :, i_pred_layer] = pred_layer
    return (out*255).astype(np.uint8)



npy = np.load(SRC)
src_height, src_width, src_channels = npy.shape

# Load numpy array and apply windowing
img_cl = np.zeros((src_height, src_width, len(SEQUENCE_WINDOWS)), dtype=np.uint8)

for i_seqtype, (seqtype_name, seqtype_dict) in enumerate(SEQUENCE_WINDOWS.items()):
    channel = seqtype_dict['channel']
    if seqtype_dict['method'] == 'window':
        wc, ww = seqtype_dict['wc'], seqtype_dict['ww']
        img_cl[:, :, i_seqtype] = window_numpy(npy[:, :, channel], wc, ww, rescale_255=True)
    elif seqtype_dict['method'] == 'divmax':
        ch = npy[:, :, channel] - npy[:, :, channel].min()
        ch = ch / ch.max()
        img_cl[:, :, i_seqtype] = (ch*255).astype(np.uint8)

# Crop and double res if needed (for HRNet which downsamples masks)
img_crop_cl, topleft_crop = center_crop(pad_if_needed(img_cl, FOV_CROP, FOV_CROP), FOV_CROP, FOV_CROP)
if DOUBLE_INPUT_RES:
    img_crop_cf = skimage.transform.rescale(img_crop_cl, 2, order=3, multichannel=True).transpose((2, 0, 1))
else:
    img_crop_cf = img_cl.transpose((2, 0, 1))

# Predict
# Load the ONNX low-res model
sess = ort.InferenceSession(MODELPATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
img_batch = np.expand_dims(img_crop_cf, 0).astype(np.float32)
pred_batch = sess.run([output_name], {input_name: img_batch})[0]  # Returns a list of len 1

# !!!HUI CAN STOP HERE!!! - Below is just for visualisation

# Plot
fig, axes = plt.subplots(2, 6, figsize=(32, 12), gridspec_kw={'wspace': 0, 'hspace': 0})

# pred coords
xs_ys_by_channel = []
for i_channel in range(pred_batch.shape[1]):  # dim 0 is batch, then prediction channels
    # pred mask
    pred = pred_batch[0, i_channel]
    if RAISE_TO_POWER > 1:
        pred = np.power(pred, RAISE_TO_POWER)
    if NORMALISE_PREDS:
        pred = pred - pred.min()
        pred = pred / (pred.max() + 1e-8)

    pred_coords = pose_mask_to_coords(prediction_mask=pred_batch[0, i_channel])
    pred_coords = [[coord['x'], coord['y']] for coord in pred_coords]
    xs, ys = zip(*pred_coords)
    xs_ys_by_channel.append([xs, ys])

for i_seqtype, (seqtype_name, seqtype_dict) in enumerate(SEQUENCE_WINDOWS.items()):

    # image
    img_seqtype = default_cmap(img_crop_cl[:, :, i_seqtype]/255)  # use _cl not _cf, as then not upsampled
    img_seqtype = (img_seqtype[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel (need to blend) & IMGify

    # Set up plot
    axes[0][i_seqtype].set_axis_off()
    axes[1][i_seqtype].set_axis_off()
    axes[0][i_seqtype].set_title(seqtype_name)

    # Draw heatmap
    img_plot = Image.fromarray(img_seqtype)
    preds_colour = preds_to_rgb(pred_batch)
    print(img_seqtype.shape, preds_colour.shape)
    img_plot = Image.blend(img_plot, Image.fromarray(preds_colour), 0.5)
    axes[0][i_seqtype].imshow(np.array(img_plot))

    # Draw base image for lines (lines added in loop)
    axes[1][i_seqtype].imshow(img_seqtype)

    for xs, ys in xs_ys_by_channel:
        # add lines to plot
        axes[1][i_seqtype].plot(xs, ys)

plt.show()

import matplotlib.pyplot as plt

for pred_channel in pred_batch[0]:
    pred_channel = pred_channel - pred_channel.min()
    pred_channel = pred_channel / (pred_channel.max() + 1e-8)

    plt.imshow(pred_channel)
    plt.title("npy")
    plt.show()

for ch in img_batch[0]:
    print(ch.min(), ch.max())

for i in range(6):
    print(np.mean(img_cl[:, :, i]))
