import cv2
import pydicom
import numpy as np
import skimage.transform
import albumentations as A
import matplotlib.pyplot as plt

import onnxruntime as ort

from utils.dataset import T1T2Dataset
from utils.vis import get_polygons


# Settings
MEAN = [0.3023, 0.4016]
STD = [0.2582, 0.3189]
FOV_LOWRES = 256
FOV_HIGHRES = 168

DIM_HIGHRES = 256  # We will scale up the 168 x 168 to 256 x 256

BORDER = 20  # Bordersize around low res detected LV


def dicom_to_img(dicom):
    if type(dicom) == str:
        dcm = pydicom.dcmread(dicom)
    else:
        dcm = dicom
    window_min = max(0, dcm.WindowCenter - dcm.WindowWidth)
    frame = dcm.pixel_array - window_min
    frame = frame / dcm.WindowWidth
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    return frame


def pad_if_needed(img, min_height, min_width):
    input_height, input_width = img.shape[:2]
    new_shape = list(img.shape)
    new_shape[0] = max(input_height, min_height)
    new_shape[1] = max(input_width, min_width)
    row_from, col_from = 0, 0
    if input_height < min_height:
        row_from = (min_height - input_height) // 2
    if input_width < min_width:
        col_from = (min_width - input_width) // 2
    out = np.zeros(new_shape, dtype=img.dtype)
    out[row_from:row_from+input_height, col_from:col_from+input_width] = img
    return out


def center_crop(img, crop_height, crop_width, centre=None):
    """Either crops by the center of the image, or around a supplied point.
    Does not pad; if the supplied centre is towards the egde of the image, the padded
    area is shifted so crops start at 0 and only go up to the max row/col

    Returns both the new crop, and the top-left coords as a row,col tuple"""
    input_height, input_width = img.shape[:2]
    if centre is None:
        row_from = (input_height - crop_height)//2
        col_from = (input_width - crop_width)//2
    else:
        row_centre, col_centre = centre
        row_from = max(row_centre - (crop_height//2), 0)
        if (row_from + crop_height) > input_height:
            row_from -= (row_from + crop_height - input_height)
        col_from = max(col_centre - (crop_width//2), 0)
        if (col_from + crop_width) > input_width:
            col_from -= (col_from + crop_width - input_width)
    return img[row_from:row_from+crop_height, col_from:col_from+crop_width], (row_from, col_from)

# Load data
# path_t1 = "../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
# path_t2 = "../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
# path_t1 = "../data/dicoms/by_date_by_study/20200505/T1SR_Mapping_SASHA_HC_T1T2_42363_622646938_622646943_2398_20200505-120210_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
# path_t2 = "../data/dicoms/by_date_by_study/20200505/T1SR_Mapping_SASHA_HC_T1T2_42363_622646938_622646943_2398_20200505-120210_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
path_t1 = "../../data/dicoms/bdbs_new/20200615/T1SR_Mapping_SASHA_HC_T1T2_42363_671978570_671978575_58_20200615-103059_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
path_t2 = "../../data/dicoms/bdbs_new/20200615/T1SR_Mapping_SASHA_HC_T1T2_42363_671978570_671978575_58_20200615-103059_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"


# LOW RES
img_t1_native = dicom_to_img(path_t1)
img_t2_native = dicom_to_img(path_t2)
input_height, input_width = img_t1_native.shape[:2]
input_pixelspacing = pydicom.dcmread(path_t1).PixelSpacing

img_t1_mm = skimage.transform.rescale(img_t1_native, input_pixelspacing, order=3)
img_t2_mm = skimage.transform.rescale(img_t2_native, input_pixelspacing, order=3)
img_mm_cl = np.dstack((img_t1_mm, img_t2_mm))
img_mm_cf = img_mm_cl.transpose((2, 0, 1))  # 2 * H * W

img_t1_lowres, topleft_lowres = center_crop(pad_if_needed(img_t1_mm, FOV_LOWRES, FOV_LOWRES), FOV_LOWRES, FOV_LOWRES)
img_t2_lowres, topleft_lowres = center_crop(pad_if_needed(img_t2_mm, FOV_LOWRES, FOV_LOWRES), FOV_LOWRES, FOV_LOWRES)

img_lowres_cl = np.dstack((img_t1_lowres, img_t2_lowres))
img_lowres_cl = (img_lowres_cl - MEAN) / STD
img_lowres_cf = img_lowres_cl.transpose((2, 0, 1))  # 2 * H * W

# Predict
# Load the ONNX low-res model
sess = ort.InferenceSession("../../output/models/016/t1t2_dlv3_lowres.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
img_batch_lowres = np.expand_dims(img_lowres_cf, 0).astype(np.float32)
pred_batch_lowres = sess.run([output_name], {input_name:img_batch_lowres})[0]  # Returns a list of len 1
pred_cls_lowres = np.argmax(pred_batch_lowres[0], axis=0)  # Classes

# Get co-ordinates for ROI
centre_row_lowres, centre_col_lowres = T1T2Dataset.get_high_res_coords_for_mask(pred_cls_lowres, get='centre')

# Crop highres region
img_highres_cl, topleft_highres = center_crop(img_lowres_cl, FOV_HIGHRES, FOV_HIGHRES, centre=(centre_row_lowres, centre_col_lowres))

# Resize the image to input size (will be needed if we want to use 256 x 256 input)
if FOV_HIGHRES != DIM_HIGHRES:
    img_highres_cl = skimage.transform.resize(img_highres_cl, (DIM_HIGHRES, DIM_HIGHRES), order=3)

img_highres_cf = img_highres_cl.transpose((2, 0, 1))  # 2 * H * W

# Predict
# Load the ONNX high-res model
sess = ort.InferenceSession("../../output/models/020/t1t2_dlv3_hires.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
img_batch_highres = np.expand_dims(img_highres_cf, 0).astype(np.float32)
pred_batch_highres = sess.run([output_name], {input_name: img_batch_highres})[0]  # Returns a list of len 1
pred_cls_highres = np.argmax(pred_batch_highres[0], axis=0)  # Classes


# VIS
fig, axes = plt.subplots(3, 2, figsize=(20,30), gridspec_kw={'wspace': 0, 'hspace': 0})

axes[0][0].imshow(img_mm_cf[0], cmap='gray')
axes[0][0].set_title("T1")
axes[0][0].set_axis_off()

axes[0][1].imshow(img_mm_cf[1], cmap='gray')
axes[0][1].set_title("T2")
axes[0][1].set_axis_off()

axes[1][0].imshow(img_lowres_cf[0], cmap='gray')
axes[1][0].set_title("T1")
axes[1][0].set_axis_off()

axes[1][1].imshow(img_lowres_cf[1], cmap='gray')
axes[1][1].set_title("T2")
axes[1][1].set_axis_off()

axes[2][0].imshow(img_highres_cf[0], cmap='gray')
axes[2][0].set_title("T1")
axes[2][0].set_axis_off()

axes[2][1].imshow(img_highres_cf[1], cmap='gray')
axes[2][1].set_title("T2")
axes[2][1].set_axis_off()

for channel, colour in zip((1, 2), ('r', 'y')):  # LV cav exterior & wall exterior
    polys = get_polygons(pred_cls_highres, channel=channel)
    style = f"{colour}-"
    for poly in polys:
        ext_x, ext_y = poly.exterior.coords.xy
        axes[2][0].plot(ext_x, ext_y, style, alpha=0.5)
        axes[2][1].plot(ext_x, ext_y, style, alpha=0.5)

        high_scale = (DIM_HIGHRES / FOV_HIGHRES)

        scaled_x_lowres = (np.array(ext_x) + (topleft_highres[1] * high_scale)) / high_scale
        scaled_y_lowres = (np.array(ext_y) + (topleft_highres[0] * high_scale)) / high_scale

        axes[1][0].plot(scaled_x_lowres, scaled_y_lowres, style, alpha=0.5)
        axes[1][1].plot(scaled_x_lowres, scaled_y_lowres, style, alpha=0.5)

        scaled_x_mm = scaled_x_lowres + topleft_lowres[1]
        scaled_y_mm = scaled_y_lowres + topleft_lowres[0]

        axes[0][0].plot(scaled_x_mm, scaled_y_mm, style, alpha=0.5)
        axes[0][1].plot(scaled_x_mm, scaled_y_mm, style, alpha=0.5)

plt.show()
