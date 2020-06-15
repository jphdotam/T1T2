import cv2
import pydicom
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

import onnxruntime as ort

from utils.dataset import T1T2Dataset
from utils.vis import get_polygons

def dicompath_to_img(dicompath):
    dcm = pydicom.dcmread(dicompath)
    window_min = max(0, dcm.WindowCenter - dcm.WindowWidth)
    frame = dcm.pixel_array - window_min
    frame = frame / dcm.WindowWidth
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    return frame

# Settings
MEAN = [0.3023, 0.4016]
STD = [0.2582, 0.3189]
HI_RES_DIM = 224
BORDER = 20  # Bordersize around low res detected LV

# Load the ONNX low-res model
sess = ort.InferenceSession("../output/models/009/t1t2_fcn.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Load data
path_t1 = "../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T1_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"
path_t2 = "../data/dicoms/by_date_by_study/20200423/T1SR_Mapping_SASHA_HC_T1T2_141613_5906470_5906478_97_20200423-102518_dicom/T2_SLC0_CON0_PHS0_REP0_SET0_AVE0_1.dcm"


# LOW RES
# Load image and create batch
img_low_cl = [dicompath_to_img(path_t1), dicompath_to_img(path_t2)]
img_low_cl = np.dstack(img_low_cl)  # H * W * 2
img_low_cl = ((img_low_cl / 255) - MEAN) / STD
img_low_cf = img_low_cl.transpose((2, 0, 1))  # 2 * H * W
img_batch_low = np.expand_dims(img_low_cf, 0).astype(np.float32)

# Forward pass & get classes for each pixel
pred_batch_low = sess.run([output_name], {input_name:img_batch_low})[0]  # Returns a list of len 1
pred_cls_low = np.argmax(pred_batch_low[0], axis=0)  # Classes

# Get co-ordinates for ROI
row_from, row_to, col_from, col_to = T1T2Dataset.get_high_res_coords_for_mask(pred_cls_low, border=20)


# HI-RES
# Load the ONNX hi-res model
sess = ort.InferenceSession("../output/models/010/t1t2_fcn_hires.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Transforms
hi_transf = A.Compose([A.LongestMaxSize(HI_RES_DIM, interpolation=cv2.INTER_CUBIC),
                       A.PadIfNeeded(min_height=HI_RES_DIM, min_width=HI_RES_DIM)])
img_hi_cl = hi_transf(image=img_low_cl[row_from:row_to, col_from:col_to])['image']
img_hi_cf = img_hi_cl.transpose((2,0,1))
img_batch_hi = np.expand_dims(img_hi_cf, 0).astype(np.float32)

# Forward pass & get classes for each pixel
pred_batch_hi = sess.run([output_name], {input_name:img_batch_hi})[0]  # Returns a list of len 1
pred_cls_hi = np.argmax(pred_batch_hi[0], axis=0)  # Classes


# VIS
fig, axes = plt.subplots(1, 2, figsize=(8,4))

axes[0].imshow(img_hi_cf[0], cmap='gray')
axes[0].set_title("T1")
axes[0].set_axis_off()

axes[1].imshow(img_hi_cf[1], cmap='gray')
axes[1].set_title("T2")
axes[1].set_axis_off()

for channel, colour in zip((1, 2), ('r', 'y')):  # LV cav exterior & wall exterior
    polys = get_polygons(pred_cls_hi, channel=channel)
    style = f"{colour}-"
    for poly in polys:
        ext_x, ext_y = poly.exterior.coords.xy
        axes[0].plot(ext_x, ext_y, style, alpha=0.5)
        axes[1].plot(ext_x, ext_y, style, alpha=0.5)

plt.show()
